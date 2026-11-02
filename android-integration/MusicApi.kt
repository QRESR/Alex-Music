package com.example.musicplayer.data

import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query

// ---------------------------------------------------------------------------
// API interfaces — only talk to YOUR server, never to YouTube directly
// ---------------------------------------------------------------------------

data class TrackMeta(
    val id: String,
    val title: String,
    val artist: String,
    val duration: Int,        // seconds
    val thumbnail: String,
    val channel: String,
    val url: String           // YouTube URL (informational only)
)

data class SearchResponse(
    val query: String,
    val results: List<TrackMeta>,
    val total: Int
)

interface MusicApi {

    /** Search tracks via our proxy server. */
    @GET("search")
    suspend fun search(
        @Query("q") query: String,
        @Query("limit") limit: Int = 20
    ): SearchResponse

    /** Get metadata for a single track. */
    @GET("track/{id}")
    suspend fun getTrack(
        @Path("id") videoId: String
    ): TrackMeta
}

// ---------------------------------------------------------------------------
// Stream URL builder — NOT an API call, just builds the URL for ExoPlayer
// ---------------------------------------------------------------------------
object StreamUrlBuilder {
    private const val BASE_URL = "https://api.example.com"  // <-- your server

    /**
     * Build the stream URL for ExoPlayer.
     * ExoPlayer will hit this URL and our server proxies the YouTube audio
     * with full Range request support.
     */
    fun streamUrl(videoId: String): String {
        return "$BASE_URL/stream/$videoId"
    }
}
