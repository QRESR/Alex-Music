"""MusicBrainz: алиасы артистов, теги, определение каверов (P2 ТЗ п.8.4/9).

Почему MusicBrainz, а НЕ Last.fm/ListenBrainz (эксперименты 2026-10-30):
  - Last.fm: требует api_key (нет ключа) → нужна ручная регистрация.
  - ListenBrainz: API 410 Gone (закрыт) → неработоспособен.
  - MusicBrainz: БЕЗ ключа, лимит 1 req/sec, кэш обязателен. ✓

Что берём:
  1. artist lookup → теги/жанры (genre) для genre_fit скоринга.
  2. artist lookup → алиасы (исп. «Miyagi» ↔ «Мияги») для artist_match.
  3. recording search → first-release-date для точного года.

Кэш: data/mb_cache.json (TTL config.MB_CACHE_TTL, 7 дней по умолчанию).
Rate-limit: не чаще config.MB_MIN_INTERVAL (1.1с) между запросами.

Использование:
    from mb_client import get_artist_tags, artist_exists, resolve_year
    tags = await get_artist_tags("Баста")     # ["russian hip hop", ...] | []
    ok = await artist_exists("Eminem")        # True/False
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from .config import config

logger = logging.getLogger("mb_client")

_BASE = "https://musicbrainz.org/ws/2"
_UA = "MusicProxy/1.0 (https://github.com/alex/music-proxy)"
_CACHE_FILE = Path(config.DATA_DIR) / "mb_cache.json"
_lock = asyncio.Lock()
_last_req = 0.0
_cache: dict | None = None


# ─── Кэш (файловый, переживает рестарт) ──────────────────────────────────
def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        if _CACHE_FILE.exists():
            _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        else:
            _cache = {}
    except Exception:  # noqa: BLE001
        _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    try:
        # чистим протухшие + ограничиваем размер (держим 5000)
        now = time.time()
        fresh = {k: v for k, v in _cache.items()
                 if isinstance(v, dict) and now - v.get("ts", 0) < config.MB_CACHE_TTL}
        if len(fresh) > 5000:
            items = sorted(fresh.items(), key=lambda kv: kv[1].get("ts", 0))[-4000:]
            fresh = dict(items)
        _cache.clear()
        _cache.update(fresh)
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.debug("mb cache save failed: %s", e)


def _cache_get(key: str):
    c = _load_cache()
    v = c.get(key)
    if isinstance(v, dict) and time.time() - v.get("ts", 0) < config.MB_CACHE_TTL:
        return v.get("d")
    return None


def _cache_set(key: str, data) -> None:
    c = _load_cache()
    c[key] = {"d": data, "ts": time.time()}
    _save_cache()


# ─── Rate-limited запрос ─────────────────────────────────────────────────
async def _mb_get(path: str, params: dict) -> dict | None:
    """GET к MusicBrainz с rate-limit (1.1с) и кэшем. None при ошибке."""
    global _last_req
    if not config.MB_ENABLED:
        return None
    cache_key = f"{path}?{sorted(params.items())}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    async with _lock:
        # ждём интервал между запросами
        global _last_req
        wait = config.MB_MIN_INTERVAL - (time.time() - _last_req)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_req = time.time()
        try:
            async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _UA}) as client:
                r = await client.get(f"{_BASE}/{path}", params={**params, "fmt": "json"})
                if r.status_code == 200:
                    data = r.json()
                    _cache_set(cache_key, data)
                    return data
                logger.info("MB HTTP %d for %s", r.status_code, path)
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning("MB request failed: %s", e)
            return None


# ─── Публичные API ───────────────────────────────────────────────────────
async def artist_lookup(artist: str, country: str = "") -> dict | None:
    """Поиск артиста: {id, name, tags:[{name,count}], ...} | None.

    country: ISO-код (RU/DE/US) для уточнения (у «Basta» есть DE-группа).
    """
    if not artist or len(artist.strip()) < 2:
        return None
    q = f'artist:"{artist.strip()}"'
    if country:
        q += f" AND country:{country}"
    data = await _mb_get("artist", {"query": q, "limit": 3})
    if not data:
        return None
    arts = data.get("artists") or []
    if not arts:
        return None
    # берём первого с наивысшим score (MB возвращает отсортированные)
    a = arts[0]
    return {
        "id": a.get("id"),
        "name": a.get("name") or "",
        "country": a.get("country") or "",
        "type": a.get("type") or "",
        "tags": [t.get("name") for t in (a.get("tags") or []) if t.get("name")][:8],
        "aliases": [
            al.get("name") for al in (a.get("aliases") or []) if al.get("name")
        ][:10],
    }


async def get_artist_tags(artist: str, country: str = "") -> list[str]:
    """Теги/жанры артиста (для genre_fit). Кэшируется. [] при ошибке.

    Пример: «Баста» → ["russian hip hop", "russian rap", "hip hop"].
    """
    if not config.MB_ENABLED:
        return []
    a = await artist_lookup(artist, country)
    if not a:
        return []
    return a.get("tags") or []


async def artist_exists(artist: str) -> bool:
    """Есть ли артист в MusicBrainz (для валидации имени)."""
    a = await artist_lookup(artist)
    return bool(a and a.get("id"))


async def artist_aliases(artist: str) -> list[str]:
    """Альтернативные имена (алиасы) — для artist_match (Miyagi ↔ Мияги)."""
    a = await artist_lookup(artist)
    if not a:
        return []
    out = [a.get("name") or ""] + (a.get("aliases") or [])
    return [x for x in out if x and x.lower() != artist.lower()][:10]


async def resolve_year(artist: str, title: str) -> int:
    """Год первого релиза записи (для свежести, если у источника нет year)."""
    if not config.MB_ENABLED or not artist or not title:
        return 0
    q = f'recording:"{title}" AND artist:"{artist}"'
    data = await _mb_get("recording", {"query": q, "limit": 1})
    if not data:
        return 0
    recs = data.get("recordings") or []
    if not recs:
        return 0
    fr = recs[0].get("first-release-date") or ""
    # формат: "2020-05-15" | "2020" | ""
    if len(fr) >= 4 and fr[:4].isdigit():
        return int(fr[:4])
    return 0


# ─── Определение кавера (ТЗ п.9) ────────────────────────────────────────
async def is_likely_cover(artist: str, title: str, original_artist_hint: str = "") -> bool:
    """Эвристика cover: артист трека ≠ артист оригинальной записи.

    Если original_artist_hint не дан — ищем recording по названию и
    сравниваем artist-credit с нашим артистом. True = похоже на кавер.

    Не используем для ЖЁСТКОГО удаления — только скоринг (ТЗ: «не удалять
    все каверы без разбора»).
    """
    if not config.MB_ENABLED or not title:
        return False
    try:
        q = f'recording:"{title}"'
        if original_artist_hint:
            q += f' AND artist:"{original_artist_hint}"'
        data = await _mb_get("recording", {"query": q, "limit": 3})
        if not data:
            return False
        recs = data.get("recordings") or []
        if not recs:
            return False
        # artist-credit первой (канонической) записи
        ac = recs[0].get("artist-credit") or []
        orig_names = " ".join((c.get("name") or "") for c in ac).lower()
        if not orig_names:
            return False
        # наш артист есть в каноне → не кавер; нет → возможно кавер
        a_l = (artist or "").lower()
        if not a_l:
            return False
        # частичное совпадение токенов
        a_toks = set(a_l.split())
        o_toks = set(orig_names.split())
        if a_toks & o_toks:
            return False
        # артист не найден среди канонических → подозрение на кавер
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("mb cover check failed: %s", e)
        return False


def stats() -> dict:
    """Для /api/metrics: размер кэша."""
    c = _load_cache()
    return {"enabled": config.MB_ENABLED, "cache_entries": len(c)}
