"""yt-dlp integration: search, metadata extraction, stream URL resolution.

Uses the yt-dlp Python API instead of subprocess for lower latency
and no process-spawning overhead.
"""

import asyncio
import json
import logging
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Optional

import yt_dlp

from .config import config
from .models import TrackMeta, StreamInfo
from .cache import search_cache, stream_cache, metadata_cache

logger = logging.getLogger("yt_dlp_client")


def _yt_dlp_extract(video_url: str, extra_opts: Optional[dict] = None) -> dict:
    """Synchronous yt-dlp extraction — called in executor."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "no_playlist": True,
        "skip_download": True,
        "extract_flat": False,
        # NO cookies: Google rotates server-side cookies within hours, which
        # killed the whole service with "Sign in to confirm you're not a bot".
        # The ANDROID innertube client gets playability OK from this IP
        # without cookies at all (verified E2E: resolve -> googlevideo 206).
        # Rate limiting + bot-block retry handle transient blocks.
        # yt-dlp enables only "deno" by default — server has node.
        "js_runtimes": dict(config.JS_RUNTIMES),
        # android: no PO Token needed for itag 18, no cookies, works from DC IP.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if extra_opts:
        opts.update(extra_opts)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        if info is None:
            raise RuntimeError(f"yt-dlp returned None for {video_url}")
        return ydl.sanitize_info(info)


# --- Rate limiting / bot-block handling -----------------------------------
# Google rate-limits bursts of player requests from one IP with a TEMPORARY
# "Sign in to confirm you're not a bot" (LOGIN_REQUIRED). It clears within
# minutes of quiet. Strategy: min gap between resolves + retry with growing
# backoff when the block is detected.
class BotBlocked(RuntimeError):
    """YouTube rate-limit ("Sign in to confirm") — transient, clears in minutes."""


_MIN_RESOLVE_GAP = 5.0       # seconds between player requests (from ONE worker)
_last_resolve_at = 0.0
_bot_block_log: list = []     # timestamps of detected blocks (for diagnostics)
# Single-flight lock: serializes resolves across concurrent requests so the
# phone's parallel track loads don't hammer YouTube (each burst of rapid
# player requests trips Google's temporary rate-limit -> "Sign in to confirm").
_resolve_lock: "asyncio.Lock | None" = None


def _get_lock() -> "asyncio.Lock":
    global _resolve_lock
    if _resolve_lock is None:
        _resolve_lock = asyncio.Lock()
    return _resolve_lock


def _is_bot_block(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "sign in to confirm" in msg or "login_required" in msg


def _throttle() -> None:
    global _last_resolve_at
    import time as _t
    delta = _t.time() - _last_resolve_at
    if delta < _MIN_RESOLVE_GAP:
        _t.sleep(_MIN_RESOLVE_GAP - delta)
    _last_resolve_at = _t.time()


def _extract_with_backoff(video_url: str, extra_opts: Optional[dict] = None) -> dict:
    """Extract with throttle + short retry on transient bot blocks.

    Deliberately short (at most one 8s nap): this runs inside an HTTP
    request, so a long backoff makes the client hang for a minute looking
    like "зависло". Long cooling is handled by the caller via Retry-After.
    """
    import time as _t
    delays = (0.0, 8.0)
    last_exc: Optional[Exception] = None
    for attempt, delay in enumerate(delays):
        if delay:
            logger.warning("bot-block, short retry after %.0fs (attempt %d)", delay, attempt + 1)
            _t.sleep(delay)
        _throttle()
        try:
            result = _yt_dlp_extract(video_url, extra_opts)
            if attempt:
                logger.info("resolve recovered after bot block (attempt %d)", attempt + 1)
            return result
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if _is_bot_block(e):
                _bot_block_log.append(_t.time())
                continue
            raise
    # Do NOT sleep the remaining 45s here — surface a typed error so the
    # route can answer fast with Retry-After instead of hanging the client.
    raise BotBlocked(f"youtube rate-limit for {video_url}: {last_exc}")


def bot_blocked_recently(window_s: int = 90) -> bool:
    """True if a bot block was seen within `window_s` seconds (for callers)."""
    import time as _t
    now = _t.time()
    return any(now - t < window_s for t in _bot_block_log[-5:])


async def _extract_info(video_url: str, extra_opts: Optional[dict] = None) -> dict:
    """Run yt-dlp in a thread pool to avoid blocking the event loop.

    Single-flight lock serializes resolves (one at a time, ~5s apart) so
    parallel track loads never burst-fire YouTube into a rate-limit.
    """
    async with _get_lock():
        loop = asyncio.get_event_loop()
        fn = partial(_extract_with_backoff, video_url, extra_opts)
        return await loop.run_in_executor(None, fn)


async def search_tracks(query: str, limit: int = 10) -> list[TrackMeta]:
    """Search YouTube for music tracks."""
    cache_key = f"search:{query.lower().strip()}:{limit}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        info = await _extract_info(
            f"ytsearch{limit}:{query}",
            extra_opts={"extract_flat": True},
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return []

    entries = info.get("entries") or []
    results = []
    for entry in entries:
        if not entry or not entry.get("id"):
            continue
        vid = entry["id"]
        thumbnails = entry.get("thumbnails") or []
        thumb_url = thumbnails[-1].get("url", "") if thumbnails else ""
        if not thumb_url:
            thumb_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"

        track = TrackMeta(
            id=vid,
            title=entry.get("title", ""),
            artist=entry.get("uploader", entry.get("channel", "")),
            duration=int(entry.get("duration") or 0),
            thumbnail=thumb_url,
            channel=entry.get("channel", entry.get("uploader", "")),
            url=entry.get("url", f"https://www.youtube.com/watch?v={vid}"),
        )
        results.append(track)

    search_cache.set(cache_key, results, config.SEARCH_CACHE_TTL)
    return results


async def get_track_metadata(video_id: str) -> Optional[TrackMeta]:
    """Get full metadata for a single track."""
    cache_key = f"meta:{video_id}"
    cached = metadata_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        info = await _extract_info(
            f"https://www.youtube.com/watch?v={video_id}"
        )
    except Exception as e:
        logger.error("Metadata fetch failed for %s: %s", video_id, e)
        return None

    thumbnails = info.get("thumbnails") or []
    thumb_url = thumbnails[-1].get("url", "") if thumbnails else ""
    if not thumb_url:
        thumb_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    track = TrackMeta(
        id=video_id,
        title=info.get("title", ""),
        artist=info.get("uploader", info.get("artist", "")),
        duration=int(info.get("duration") or 0),
        thumbnail=thumb_url,
        channel=info.get("channel", info.get("uploader", "")),
        url=f"https://www.youtube.com/watch?v={video_id}",
    )

    metadata_cache.set(cache_key, track, config.METADATA_CACHE_TTL)
    return track


async def resolve_stream_url(video_id: str) -> StreamInfo:
    """Resolve the best audio stream URL for a video.

    Returns the CDN URL + http_headers needed for authentication.
    YouTube headers contain cookies/tokens that MUST be forwarded.
    """
    cache_key = f"stream:{video_id}"
    cached = stream_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        info = await _extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            extra_opts={
                # "web" client exposes only itag 18 (mp4 with m4a audio)
                # — bestaudio does not exist for this client.
                "format": "18/best",
            },
        )
    except Exception as e:
        raise RuntimeError(f"Failed to resolve stream for {video_id}: {e}")

    # In modern yt-dlp, after format selection the resolved URL and headers
    # live at the TOP level of info. info["format"] is a human-readable
    # string like '251 - audio only (medium)', NOT a format dict.
    selected_format = info.get("format")
    if not isinstance(selected_format, dict):
        selected_format = {}
    direct_url = selected_format.get("url") or info.get("url")
    if not direct_url:
        raise RuntimeError(f"No stream URL found for {video_id}")

    # http_headers MUST be forwarded — contains auth cookies/tokens
    http_headers = (
        selected_format.get("http_headers")
        or info.get("http_headers")
        or {}
    )

    # Parse expire time from URL for accurate TTL
    expires_at = time.time() + config.STREAM_URL_CACHE_TTL
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(direct_url)
        params = parse_qs(parsed.query)
        if "expire" in params:
            url_expire = int(params["expire"][0])
            # Use 90% of actual expiry for safety margin
            expires_at = min(expires_at, time.time() + (url_expire - int(time.time())) * 0.9)
    except Exception:
        pass

    stream = StreamInfo(
        video_id=video_id,
        direct_url=direct_url,
        http_headers=http_headers,
        expires_at=expires_at,
    )

    # Cache until URL expires (with safety margin)
    cache_ttl = max(60, int(expires_at - time.time()))
    stream_cache.set(cache_key, stream, cache_ttl)
    return stream
