package com.example.musicplayer.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.musicplayer.data.ApiClient
import com.example.musicplayer.data.TrackMeta
import com.example.musicplayer.player.MusicPlayer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel that ties search + player together.
 * All network calls go through our proxy server.
 */
class MusicViewModel : ViewModel() {

    private val api = ApiClient.musicApi

    // Search state
    private val _searchResults = MutableStateFlow<List<TrackMeta>>(emptyList())
    val searchResults: StateFlow<List<TrackMeta>> = _searchResults.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private val _searchError = MutableStateFlow<String?>(null)
    val searchError: StateFlow<String?> = _searchError.asStateFlow()

    fun search(query: String) {
        if (query.isBlank()) return
        viewModelScope.launch {
            _isSearching.value = true
            _searchError.value = null
            try {
                val response = api.search(q = query, limit = 20)
                _searchResults.value = response.results
            } catch (e: Exception) {
                _searchError.value = e.message ?: "Search failed"
                _searchResults.value = emptyList()
            } finally {
                _isSearching.value = false
            }
        }
    }

    /**
     * Play a track. MusicPlayer builds the stream URL pointing
     * to our server — ExoPlayer never sees YouTube.
     */
    fun playTrack(player: MusicPlayer, track: TrackMeta) {
        player.play(track)
    }

    fun clearSearch() {
        _searchResults.value = emptyList()
        _searchError.value = null
    }
}
