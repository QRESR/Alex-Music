"""Persistent playlist storage (JSON file)."""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import config

_DATA_DIR = Path(config.DATA_DIR)
_DATA_FILE = _DATA_DIR / "playlists.json"

_lock = threading.Lock()


def _load() -> dict:
    if not _DATA_FILE.exists():
        return {"playlists": []}
    try:
        return json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"playlists": []}


def _save(store: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DATA_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(_DATA_FILE)


def list_playlists() -> list[dict]:
    with _lock:
        return _load()["playlists"]


def get_playlist(playlist_id: str) -> Optional[dict]:
    with _lock:
        for p in _load()["playlists"]:
            if p["id"] == playlist_id:
                return p
    return None


def create_playlist(name: str) -> dict:
    playlist = {
        "id": uuid.uuid4().hex[:12],
        "name": name.strip() or "Без названия",
        "tracks": [],
        "created_at": int(time.time()),
    }
    with _lock:
        store = _load()
        store["playlists"].append(playlist)
        _save(store)
    return playlist


def delete_playlist(playlist_id: str) -> bool:
    with _lock:
        store = _load()
        before = len(store["playlists"])
        store["playlists"] = [
            p for p in store["playlists"] if p["id"] != playlist_id
        ]
        changed = len(store["playlists"]) != before
        if changed:
            _save(store)
        return changed


def add_track(playlist_id: str, track: dict) -> Optional[dict]:
    with _lock:
        store = _load()
        for p in store["playlists"]:
            if p["id"] == playlist_id:
                if not any(t["id"] == track["id"] for t in p["tracks"]):
                    p["tracks"].append(track)
                    _save(store)
                return p
    return None


def remove_track(playlist_id: str, track_id: str) -> Optional[dict]:
    with _lock:
        store = _load()
        for p in store["playlists"]:
            if p["id"] == playlist_id:
                p["tracks"] = [t for t in p["tracks"] if t["id"] != track_id]
                _save(store)
                return p
    return None
