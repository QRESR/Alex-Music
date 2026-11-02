"""Local listening history + smart "что послушать" suggestions.

Everything is stored in data/history.json (atomic writes, thread-safe):

    {
      "meta": {
        "<video_id>": {
          "id","title","artist","duration","thumbnail","channel","url",
          "plays": n, "last_play": ts, "seen": n, "last_seen": ts
        },
        ...
      },
      "searches": [ {"q": "...", "ts": ..., "n": 1}, ... ]
    }

- record_search() learns which tracks came back for which queries (and keeps
  their metadata, so later plays can be titled).
- record_play() bumps the play counter for /stream/{id} requests.
- suggestions() ranks tracks by plays + recency, so "что послушать" returns
  what the user actually listens to instead of a literal YouTube search for
  the phrase "что послушать" (which used to return audiobooks).

Deliberately offline: no network, no LLM — just counters over data we already
served.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .config import config

_DATA_DIR = Path(config.DATA_DIR)
_DATA_FILE = _DATA_DIR / "history.json"

_lock = threading.Lock()

_MAX_META = 1000       # keep metadata for at most N tracks
_MAX_SEARCHES = 300    # query log length


def _empty() -> dict:
    return {"meta": {}, "searches": []}


def _load() -> dict:
    if not _DATA_FILE.exists():
        return _empty()
    try:
        data = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty()
        data.setdefault("meta", {})
        data.setdefault("searches", [])
        return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def _save(store: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DATA_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tmp.replace(_DATA_FILE)


def _track_fields(track: dict) -> dict:
    """Keep only TrackMeta-compatible fields (counters live alongside)."""
    g = track.get("genres") or []
    return {
        "id": track.get("id", ""),
        "title": track.get("title", ""),
        "artist": track.get("artist", ""),
        "duration": int(track.get("duration") or 0),
        "thumbnail": track.get("thumbnail", ""),
        "channel": track.get("channel", ""),
        "url": track.get("url", ""),
        # ТЗ п.3: жанры (из Яндекса) — для предпочитаемых жанров профиля
        "genres": [str(x) for x in g if x][:8],
    }


def record_search(query: str, tracks: list[dict]) -> None:
    """Remember a successful search: query + metadata of returned tracks."""
    q = (query or "").strip().lower()
    now = int(time.time())
    with _lock:
        store = _load()
        if q:
            searches = store["searches"]
            for s in searches:
                if s.get("q") == q:
                    s["n"] = int(s.get("n", 1)) + 1
                    s["ts"] = now
                    break
            else:
                searches.append({"q": q, "ts": now, "n": 1})
            if len(searches) > _MAX_SEARCHES:
                searches.sort(key=lambda s: s.get("ts", 0))
                del searches[: len(searches) - _MAX_SEARCHES]
        meta = store["meta"]
        changed = False
        for t in tracks:
            tid = (t or {}).get("id")
            if not tid:
                continue
            entry = meta.get(tid)
            fields = _track_fields(t)
            if entry is None:
                meta[tid] = {**fields, "plays": 0, "last_play": 0,
                             "seen": 1, "last_seen": now}
                changed = True
            else:
                # refresh metadata (titles/thumbnails drift), count exposure
                entry.update({k: v for k, v in fields.items() if v})
                entry["seen"] = int(entry.get("seen", 0)) + 1
                entry["last_seen"] = now
                changed = True
        if changed:
            _prune(store)
            _save(store)


def record_tracks_meta(tracks: list[dict]) -> None:
    """Learn track metadata without registering a user query (fallback fb)."""
    record_search("", tracks)


def record_play(video_id: str) -> None:
    """Count a play (called from /stream, throttled by the caller)."""
    if not video_id:
        return
    with _lock:
        store = _load()
        meta = store["meta"]
        entry = meta.get(video_id)
        now = int(time.time())
        if entry is None:
            meta[video_id] = {
                "id": video_id, "title": "", "artist": "", "duration": 0,
                "thumbnail": "", "channel": "", "url": "",
                "plays": 1, "last_play": now, "seen": 0, "last_seen": now,
            }
        else:
            entry["plays"] = int(entry.get("plays", 0)) + 1
            entry["last_play"] = now
        _prune(store)
        _save(store)


def recent(limit: int = 20) -> list[dict]:
    """ТЗ п.3/п.4: последние ПРОСЛУШАННЫЕ треки (last_play desc).

    «Что играло недавно» для контекста LLM и «продолжи в том же духе».
    Возвращает TrackMeta-словари со счётчиков.
    """
    with _lock:
        store = _load()
        meta = store["meta"]
        played = [e for e in meta.values() if int(e.get("last_play", 0)) > 0]
        played.sort(key=lambda e: int(e.get("last_play", 0)), reverse=True)
    out: list[dict] = []
    for e in played[:limit]:
        f = _track_fields(e)
        f["id"] = e.get("id", "")
        out.append(f)
    return out


def last_searches(limit: int = 5) -> list[dict]:
    """ТЗ п.4: последние поисковые запросы (для контекста диалога)."""
    with _lock:
        searches = sorted(_load()["searches"], key=lambda s: int(s.get("ts", 0)),
                          reverse=True)
    return [{"q": s.get("q", ""), "ts": s.get("ts", 0)} for s in searches[:limit]]


def track_meta(track_id: str) -> dict:
    """Метаданные трека из истории (включая genres) — для профиля вкуса."""
    if not track_id:
        return {}
    with _lock:
        e = _load()["meta"].get(track_id)
    return dict(e) if e else {}


def _prune(store: dict) -> None:
    """Drop the least valuable meta entries when the file grows too big."""
    meta = store["meta"]
    if len(meta) <= _MAX_META:
        return
    now = time.time()

    def score(e: dict) -> float:
        plays = int(e.get("plays", 0)) * 100
        fresh = max(0.0, 7 * 86400 - (now - max(int(e.get("last_play", 0)),
                                                  int(e.get("last_seen", 0)))))
        return plays + fresh

    ranked = sorted(meta.items(), key=lambda kv: score(kv[1]), reverse=True)
    store["meta"] = dict(ranked[:_MAX_META])


def _score(entry: dict, now: float) -> float:
    plays = int(entry.get("plays", 0))
    seen = int(entry.get("seen", 0))
    last = max(int(entry.get("last_play", 0)), int(entry.get("last_seen", 0)))
    age = now - last if last else 10 ** 9
    if age < 86400:
        fresh = 40.0
    elif age < 7 * 86400:
        fresh = 20.0
    elif age < 30 * 86400:
        fresh = 8.0
    else:
        fresh = 0.0
    return plays * 10.0 + min(seen, 10) * 2.0 + fresh


def suggestions(limit: int = 30) -> list[dict]:
    """Rank known tracks: most played + recently seen first.

    Returns TrackMeta-shaped dicts (counters stripped) — titles come from
    tracks we previously returned in search results.
    """
    with _lock:
        store = _load()
        meta = store["meta"]
        now = time.time()
        ranked = sorted(
            (
                (score, eid, e)
                for eid, e in meta.items()
                if eid and (e.get("title") or e.get("plays") or e.get("seen"))
                for score in (_score(e, now),)
            ),
            key=lambda row: row[0],
            reverse=True,
        )
    out: list[dict] = []
    for score, eid, entry in ranked[:limit]:
        if not entry.get("title"):
            # never-titled ids (played without a known search hit) are useless
            # to a UI — skip unless they have plays AND a url to identify them
            if int(entry.get("plays", 0)) <= 0:
                continue
        fields = _track_fields(entry)
        fields["id"] = eid
        out.append(fields)
    return out
