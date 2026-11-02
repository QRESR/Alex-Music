package com.example.musicplayer.player

import android.content.Context
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import com.example.musicplayer.data.StreamUrlBuilder
import com.example.musicplayer.data.TrackMeta
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Music player that streams through our proxy server.
 *
 * Key points:
 * - ExoPlayer hits ONLY our server URLs (never YouTube directly)
 * - Range requests are handled automatically by ExoPlayer + our proxy
 * - DataSource factory handles cross-protocol redirects and timeouts
 *
 * ExoPlayer stream flow:
 *   1. HEAD /stream/{id}     → gets Content-Length, Accept-Ranges
 *   2. GET  /stream/{id}     → initial connection
 *   3. GET  /stream/{id} Range: bytes=X-Y  → seek/buffer
 */
@UnstableApi
class MusicPlayer(private val context: Context) {

    private var exoPlayer: ExoPlayer? = null

    private val _currentState = MutableStateFlow(PlayerState())
    val currentState: StateFlow<PlayerState> = _currentState.asStateFlow()

    data class PlayerState(
        val isPlaying: Boolean = false,
        val currentTrack: TrackMeta? = null,
        val progress: Long = 0,
        val duration: Long = 0,
        val buffered: Long = 0,
        val error: String? = null
    )

    fun initialize() {
        // Configure DataSource for proxy server
        val dataSourceFactory = DefaultHttpDataSource.Factory()
            .setConnectTimeoutMs(15_000)
            .setReadTimeoutMs(30_000)
            .setUserAgent("MusicPlayer/1.0 (Android)")
            .setAllowCrossProtocolRedirects(true)  // HTTP→HTTPS redirects

        exoPlayer = ExoPlayer.Builder(context)
            .setMediaSourceFactory(
                DefaultMediaSourceFactory(dataSourceFactory)
            )
            .setHandleAudioBecomingNoisy(true)
            .setWakeMode(C.WAKE_MODE_NETWORK)
            .build()
            .apply {
                addListener(playerListener)
            }
    }

    /**
     * Play a track through our proxy server.
     * The ONLY network call ExoPlayer makes is to:
     *   https://api.example.com/stream/{videoId}
     *
     * Our server handles:
     * - Resolving the YouTube stream URL via yt-dlp
     * - Forwarding Range requests for seeking
     * - Caching the resolved URL
     */
    fun play(track: TrackMeta) {
        val player = exoPlayer ?: return

        val streamUrl = StreamUrlBuilder.streamUrl(track.id)

        val mediaItem = MediaItem.Builder()
            .setMediaId(track.id)
            .setUri(streamUrl)
            .setMediaMetadata(
                MediaMetadata.Builder()
                    .setTitle(track.title)
                    .setArtist(track.artist)
                    .setArtworkUri(track.thumbnail.toUriOrNull())
                    .build()
            )
            .build()

        player.setMediaItem(mediaItem)
        player.prepare()
        player.playWhenReady = true

        _currentState.value = _currentState.value.copy(
            currentTrack = track,
            error = null
        )
    }

    fun pause() { exoPlayer?.playWhenReady = false }
    fun resume() { exoPlayer?.playWhenReady = true }
    fun seekTo(positionMs: Long) { exoPlayer?.seekTo(positionMs) }
    fun stop() { exoPlayer?.stop() }

    fun release() {
        exoPlayer?.removeListener(playerListener)
        exoPlayer?.release()
        exoPlayer = null
    }

    private val playerListener = object : Player.Listener {
        override fun onIsPlayingChanged(isPlaying: Boolean) {
            _currentState.value = _currentState.value.copy(isPlaying = isPlaying)
        }

        override fun onPlaybackStateChanged(playbackState: Int) {
            val player = exoPlayer ?: return
            when (playbackState) {
                Player.STATE_READY -> {
                    _currentState.value = _currentState.value.copy(
                        duration = player.duration.coerceAtLeast(0),
                        buffered = player.bufferedPosition,
                        error = null
                    )
                }
                Player.STATE_BUFFERING -> {
                    _currentState.value = _currentState.value.copy(
                        buffered = player.bufferedPosition
                    )
                }
                Player.STATE_ENDED -> {
                    _currentState.value = _currentState.value.copy(isPlaying = false)
                }
            }
        }

        override fun onPlayerError(error: PlaybackException) {
            _currentState.value = _currentState.value.copy(
                error = error.message ?: "Playback error",
                isPlaying = false
            )
        }
    }

    private fun String.toUriOrNull() = try {
        android.net.Uri.parse(this)
    } catch (_: Exception) { null }
}
