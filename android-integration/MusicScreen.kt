package com.example.musicplayer.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.musicplayer.data.TrackMeta
import kotlin.math.floor

/**
 * Example Compose screen — search + track list + player controls.
 * All data flows through our proxy server.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MusicScreen(viewModel: MusicViewModel) {
    var query by remember { mutableStateOf("") }
    val results by viewModel.searchResults.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()
    val error by viewModel.searchError.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Search bar
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Search music...") },
            singleLine = true
        )

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = { viewModel.search(query) },
            modifier = Modifier.fillMaxWidth(),
            enabled = query.isNotBlank() && !isSearching
        ) {
            if (isSearching) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text("Search")
        }

        // Error
        error?.let {
            Spacer(modifier = Modifier.height(8.dp))
            Text(it, color = MaterialTheme.colorScheme.error)
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Results
        LazyColumn(modifier = Modifier.fillMaxSize()) {
            items(results) { track ->
                TrackItem(track = track, onClick = { /* play track */ })
            }
        }
    }
}

@Composable
fun TrackItem(track: TrackMeta, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Thumbnail
        AsyncImage(
            model = track.thumbnail,
            contentDescription = null,
            modifier = Modifier
                .size(56.dp)
                .padding(end = 12.dp),
            contentScale = ContentScale.Crop
        )

        // Info
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = track.title,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1
            )
            Text(
                text = track.artist.ifEmpty { track.channel },
                style = MaterialTheme.typography.bodySmall,
                maxLines = 1
            )
        }

        // Duration
        if (track.duration > 0) {
            val min = floor(track.duration / 60.0).toInt()
            val sec = track.duration % 60
            Text(
                text = "%d:%02d".format(min, sec),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
