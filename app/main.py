"""FastAPI application — Alex Music server.

Endpoints:
    GET/HEAD /stream/{id}        — audio stream (Range proxy)
    GET       /search?q=...      — search tracks (Yandex / VK)
    GET       /track/{id}        — track metadata
    GET       /health            — healthcheck

CRITICAL for ExoPlayer:
    - HEAD support (ExoPlayer probes with HEAD first)
    - Accept-Ranges: bytes on ALL responses
    - Content-Length on ALL responses
    - 206 Partial Content for Range requests
    - Proper Content-Range header
"""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response, Header
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic import BaseModel as _BM  # alias для роутов /smart, /api/applog

from .config import config
from .models import TrackMeta, SearchResponse, ErrorResponse
from .yt_dlp_client import (
    search_tracks,
    get_track_metadata,
    resolve_stream_url,
    BotBlocked,
    bot_blocked_recently,
)
from .cache import search_cache, stream_cache, metadata_cache
from . import playlists as playlist_store
from .intent import parse_intent
from . import llm

logger = logging.getLogger("music_proxy")
# INFO от наших модулей (music_proxy/ym_client/ma_client) иначе НЕ пишутся:
# root по умолчанию WARNING, basicConfig нигде не вызывался → слепая диагностика.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _name in ("music_proxy", "ym_client", "ma_client", "yt_dlp_client"):
    logging.getLogger(_name).setLevel(logging.INFO)


def parse_range_header(range_header: str, total_size: int) -> tuple[int, int]:
    """Parse 'bytes=0-' or 'bytes=100-200' into (start, end)."""
    try:
        ranges = range_header.replace("bytes=", "").split("-")
        start = int(ranges[0]) if ranges[0] else 0
        end = int(ranges[1]) if ranges[1] else total_size - 1
        # Clamp to valid range
        start = max(0, min(start, total_size - 1))
        end = max(start, min(end, total_size - 1))
        return start, end
    except (ValueError, IndexError):
        raise HTTPException(status_code=416, detail="Invalid Range header")


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------
async def _periodic_cleanup():
    """Evict expired cache entries every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        search_cache.cleanup()
        stream_cache.cleanup()
        metadata_cache.cleanup()
        logger.info(
            "Cache cleanup: search=%d stream=%d meta=%d",
            len(search_cache), len(stream_cache), len(metadata_cache),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_periodic_cleanup())
    logger.info("Alex Music server started")
    yield
    task.cancel()
    logger.info("Alex Music server stopped")


app = FastAPI(
    title="Alex Music",
    version="1.0.0",
    lifespan=lifespan,
)

# Shared httpx client for proxying (connection pooling)
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        # CRITICAL: YouTube CDN URLs embed the client IP yt-dlp detected.
        # yt-dlp sees the server's IPv6 address, so requests MUST go over IPv6.
        # Over IPv4 the IP mismatches and googlevideo returns 403/302.
        # local_address="::" forces AF_INET6 sockets.
        transport = httpx.AsyncHTTPTransport(local_address="::")
        _http_client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            timeout=httpx.Timeout(config.PROXY_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )
    return _http_client


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------
def _to_mix_query(q: str) -> str:
    """Turn a base query into a 'related / mix' query for a set of tracks.

    YouTube treats "<X> mix/playlist/radio" as a rich related set — this is
    how we deliver 'подборку/mix' semantics without a dedicated endpoint.
    """
    return f"{q} mix".strip()


def _is_pure_genre(text: str) -> bool:
    """True, если запрос — ЧИСТЫЙ жанр («рэп», «хип-хоп», «джаз») без ситуации.

    Жанр ищем ДОСЛОВНО (LLM-перефраз «рэп» → «русский рэп для тренировки»
    даёт мусор). Ситуации («для дороги») идут через mood_mix_query как раньше.
    Слова-мусор (включи/подборка/музыка/треки) отбрасываем перед сравнением.
    """
    t = re.sub(
        r"\b(включи|включить|включим|поставь|поиграй|дай|найди|подборк\w*|микс|"
        r"плейлист\w*|музык\w*|песн\w*|трек\w*|свеж\w*|нов\w*|пацалуй|пожалуйста)\b",
        " ", (text or "").lower(),
    )
    t = re.sub(r"[^0-9a-zа-яё\- ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    words = t.split()
    # каждый токен должен быть жанром (или его частью: «хип-хоп», «рэп»)
    from .intent import _GENRE_WORDS

    return all(any(g in w or w in g for g in _GENRE_WORDS) for w in words)


# ——— СПАМ-ARTIST (2026-10-31, жалоба «косяки по именам/названию группы») ———
# Яндекс кладёт в artist целый бред-тайтл: на «баста без слим шейди»
# подстрочный якорь матчил «Thiagão - Só Tua Graça Me Basta (2014)
# rappratodos.blogspot.com.br ...» как артиста → 5 из 6 выдачи — бразильцы.
# Настоящий artist короче и без ссылок/годов: «Jay-Z, Linkin Park - Big
# Pimpin' / Papercut» = 43 симв.
_JUNK_ARTIST_RE = re.compile(
    r"https?://|www\.|blogspot|\.(?:com|net|org)\b|\((?:19|20)\d{2}\)|"
    r"rappratodos",
    re.I,
)


def _artist_is_spam(a: str) -> bool:
    """True, если поле artist — спам-ряд (ссылки/годы/длинный бред)."""
    return len(a) > 60 or bool(_JUNK_ARTIST_RE.search(a))


@app.get("/search", response_model=SearchResponse)
async def api_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    mix: int = Query(0, ge=0, le=1, description="1 = playlist/mix (related set)"),
    ctx: str = Query("", description="Текущий трек («артист - название») для подборки по настроению"),
    source: str = Query("auto", description="auto|yandex|vk — какой источник основной (для переключателя в приложении)"),
):
    """Search tracks (Yandex/VK), with optional Alice-like intent handling.

    mix=1 -> return a related SET (playlist/radio) instead of a literal match.
    The response carries `intent` so the client knows whether to autoplay a
    single hit or fill a queue.
    """
    stored = playlist_store.list_playlists()
    it = parse_intent(q, stored_playlists=stored)

    # A named stored playlist wins outright — no network needed.
    if it.matched_playlist_id:
        pl = playlist_store.get_playlist(it.matched_playlist_id)
        if pl is not None:
            from .models import TrackMeta as _TM

            tracks = [_TM(**t) for t in pl.get("tracks", [])]
            return SearchResponse(
                query=q,
                results=tracks[: config.MAX_SEARCH_RESULTS],
                total=len(tracks),
                # piggy-back intent on the response via a wrapper below
            )

    try:
        from . import metrics as _mt_s
        _mt_s.event("search_total")
        if not tracks:
            _mt_s.event("search_empty")
    except Exception:  # noqa: BLE001
        pass
    effective_mix = 1 if (mix == 1 or it.wants_many) else 0
    search_q = _to_mix_query(it.query) if effective_mix else (it.query or q)
    # ЛИМИТ ВЫДАЧИ (2026-11-01, просьба «в поиске не больше 4 треков»):
    # обычный поиск → 4 («4 штуки хватит чтобы выбрать»); подборка/mix → 10
    # Поиск = 10 (2026-11-01: пользователь отменил лимит-4 — «пусть 10 там,
    # чтобы выбирать было»; после 1-го трека переход в похожий, но список из 10
    # нужен для выбора). Подборка = 20 (пользователь: «подборку из 10 треков»
    # — фактически отдаём до MAX_SEARCH_RESULTS). Режем независимо от limit
    # APK (10/20) — это верхняя граница, не default.
    effective_limit = min(limit, config.MAX_SEARCH_RESULTS)

    # ---- LLM «по настроению»: только для ПОДБОРОК (intent=playlist/wants_many).
    # Иначе LLM запускается на обычном поиске («playlist lullaby» → play) и
    # mood_query ломает точный поиск. Смысл: из просьбы и текущего трека (ctx)
    # LLM выведет связный запрос («колыбельные для сна»), Яндекс ищет набор.
    mood_query: str | None = None
    mood_base = it.query or q

    # ФИКС «песни из сериала X → мусорная подборка» (2026-11-01, жалоба:
    # «песни из очень странные дела»): паттерн «песни/саундтрек/музыка из
    # [сериала/фильма] X» = ТОЧНОЕ название X, а не артист и не жанр. Без фикса:
    # wants_many(«песни») → effective_mix → mode=mood → fallback-якорь берёт
    # ПОСЛЕДНЕЕ слово («дела») как артист → Яндекс ищет «дела» → мусор
    # (live 11:39: Michael Jackson/Сова/Комбинация, match@5=0.00; ВК спасался
    # только своим аудио-поиском). Решение: детерминированно переписываем в
    # «{X} OST» и ВЫКЛЮЧАЕМ подборку — это точный запрос.
    # ЖИВЫЕ ВАРИАНТЫ: «песни из сериала очень странные дела» / «песни из очень
    # странные дела» (без «сериала»). Маркер сериала опционален; без него
    # требуем ≥2 длинных слова («песни из 90-х»/«из кино» — не ловим, это набор).
    _mq = re.search(
        r"^(?:песн\w*|саундтрек\w*|музык\w*|звук\w*|хит\w*|трек\w*)\s+из\s+"
        r"(?:(сериала|мультсериала|фильм\w*|кинофильм\w*|мультфильм\w*|"
        r"игр\w*|концерт\w*|аниме|цикла|шоу)\s+)?(.+)$",
        (it.query or q or "").strip(), re.I)
    _media_title = ""
    if _mq:
        _has_marker = bool(_mq.group(1))
        _media_title = _mq.group(2).strip()
        # убираем хвостовые маркеры («... саундтрек», «... ost») если юзер повторил
        _media_title = re.sub(
            r"[\s(]*(?:саундтрек\w*|ост|ost|soundtrack)[\s)]*$",
            "", _media_title, flags=re.I).strip(" .,!?;:«»\"'")
        # без явного маркера («сериала») — только если ≥2 длинных слова и нет цифр
        if not _has_marker:
            _mw = [w for w in re.split(r"\s+", _media_title) if len(w) >= 3]
            if len(_mw) < 2 or re.search(r"\d", _media_title):
                _media_title = ""
    if _media_title and len(_media_title) >= 3:
        _orig_q = q or ""
        it.query = f"{_media_title} OST"
        # wants_many — property от intent/is_playlist (нет сеттера) — меняем поля:
        if it.intent == "playlist" or it.is_playlist:
            it.intent = "play"      # из INTENT_PLAYLIST убираем — точный запрос
            it.is_playlist = False
        mood_base = it.query
        # ВАЖНО: effective_mix/search_q уже посчитаны ВЫШЕ (line ~216-217)
        # с wants_many=True — пересчитываем, иначе подборка уже включена.
        effective_mix = 0
        search_q = it.query
        logger.info(
            "media-title фикс: %r -> %r (exact, не подборка)",
            _orig_q[:60], it.query,
        )

    want_mood = bool(
        it.wants_many
        or effective_mix
        or re.search(r"по настроен|настроен|похож|другой|другого|не этого|подбери", mood_base, re.I)
    )
    # ФИКС «жанр → ерунда»: чистый ЖАНРОВОЙ запрос БЕЗ подборки («рэп», «хип-хоп»)
    # НЕ отдаём LLM — она придумывала отсебятину («русский рэп для тренировки»)
    # → мусор. НО «рэп ПОДБОРКА» / «танцевальная ПОДБОРКА» = явный запрос НАБОРА
    # → LLM нужна, чтобы построить связный жанровый запрос (иначе Яндекс ищет
    # буквально и отдаёт треки с НАЗВАНИЕМ «Танцевальная» — журнал 09:37).
    _wants_set_word = bool(
        re.search(r"подборк|микс|плейлист|сборник|что-нибудь|что нибудь", mood_base, re.I)
    )
    is_pure_genre = bool(it.wants_many and _is_pure_genre(mood_base) and not _wants_set_word)

    # P1 MusicIntent (ТЗ 4.4): СТРУКТУРИРОВАННЫЙ разбор вместо одной строки.
    # LLM возвращает {mood, activity, energy, genres, freshness, seed_artist, ...},
    # из которого сервер строит НЕСКОЛЬКО поисковых запросов (multi-strategy).
    # Только при USE_NEW_RANKER (патч) — иначе старый mood_mix_query как раньше.
    music_intent = None
    # ТЗ п.6: разбор НЕ только для подборок — точный запрос с явным
    # ИСКЛЮЧЕНИЕМ («баста без слим шейди») требует exclude_artists.
    # Один дополнительный LLM-вызов оправдан: маркер явный (п.10).
    _want_exclude = bool(re.search(
        r"(^|\s)(без\s|кроме\s|не\s+включай|только\s+не|не\s+хочу|исключая)",
        (q or "").lower()))
    if (want_mood or _want_exclude) and not is_pure_genre and config.USE_NEW_RANKER:
        try:
            from . import music_intent as _mi
            music_intent = await _mi.parse_intent_llm(mood_base, current_track=ctx)
            if music_intent is None:
                # LLM недоступен/мусор → детерминированный offline-парсер
                music_intent = _mi.parse_intent_offline(mood_base)
        except Exception as e:  # noqa: BLE001
            logger.warning("music_intent failed: %s", e)
            music_intent = None

    # ТЗ п.6: LLM задаёт РЕЖИМ — сервер исполняет. mode=playlist → подборка,
    # mode=exact → точный поиск (поверх regex; якорь по пулу в /search —
    # страховка, если LLM ошиблась). Пустой mode не трогает ничего.
    if music_intent is not None:
        if music_intent.mode == "playlist" and not effective_mix:
            effective_mix = 1
            logger.info("music intent mode=playlist -> mix=1")
        elif music_intent.mode == "exact" and effective_mix:
            effective_mix = 0
            mood_query = None
            logger.info("music intent mode=exact -> mix=0")

    if want_mood and not is_pure_genre:
        if music_intent is not None and music_intent.action in ("play_music", "search"):
            # из intent строим поисковую строку (LLM дал готовый query — берём)
            refined = music_intent.query or ""
            if not refined:
                qs = music_intent.to_search_queries()
                refined = qs[0] if qs else ""
            if refined and refined.lower() != mood_base.lower():
                mood_query = refined
        else:
            # фолбэк: старый текстовый mood_mix_query (если intent не собрали)
            try:
                refined = await llm.mood_mix_query(mood_base, current_track=ctx)
                if refined and refined.lower() != mood_base.lower():
                    mood_query = refined
            except Exception as e:  # noqa: BLE001
                logger.warning("mood_mix_query failed: %s", e)

    # ФИКС «жанр-подборка → одноимённые треки»: если LLM не дала СВЯЗНОГО
    # запроса (query == исходный «танцевальная подборка», либо intent None),
    # строим жанр-якорь ДЕТЕРМИНИРОВАННО: «танцевальная подборка» →
    # «танцевальная подборка танцевальный микс». Без этого Яндекс ищет
    # буквально и отдаёт треки с НАЗВАНИЕМ «Танцевальная» (журнал 09:37).
    if want_mood and effective_mix and not mood_query:
        from . import intent as _it_genre
        _genre_words = [w for w in (mood_base or "").split()
                        if any(g in w for g in _it_genre._GENRE_WORDS)]
        if _genre_words:
            _genre_anchor = " ".join(_genre_words)
            _qualifiers = ("лучшие", "популярные", "танцевальный микс",
                           "для вечеринки", "подборка")
            # избегаем дубля «подборка подборка»
            _extra = [w for w in _qualifiers if w not in (mood_base or "")]
            mood_query = f"{_genre_anchor} {' '.join(_extra[:2])}".strip()
            logger.info("genre playlist anchor: %r -> mood_query=%r",
                        mood_base, mood_query)

    # ---- SOURCE ROUTING (переключатель в приложении: auto|yandex|vk) -------
    # source=yandex: только Яндекс + MA fallback (без VK)
    # source=vk: только ВК (без Яндекса/MA)
    # source=auto: текущая логика (Яндекс → VK → MA)
    tracks: list = []

    # ---- CANDIDATE GENERATION, параллельная (ТЗ п.3.2/п.15) ---------------
    # VK и MA стартуем СРАЗУ (одновременно с Яндексом), а не ждём по цепочке:
    # итоговое время = max(источники), а не sum. Merge-приоритет сохраняем
    # прежний (Яндекс → VK → MA).
    # ВК параллелится только когда нужен большой пул (подборки/mood) или
    # source=vk — точный поиск НЕ дёргает ВК зря (флуд-контроль VK API).
    from . import ma_client as _mac_global
    from . import vk_client as _vkc_global

    async def _safe(fn, *args):
        """Задача-источник: ошибка источника → пусто (не роняет выдачу).

        ТЗ п.1: каждый источник идёт под общим семафором (rate-limit RAM/сеть)
        + свой таймаут (внутри клиентов: ym 20с, vk 25-45с, ma 10с).
        """
        from . import concurrency as _cc
        try:
            async with _cc.source_gate():
                return await fn(*args)
        except Exception as e:  # noqa: BLE001
            logger.warning("candidate source failed: %s: %s", getattr(fn, "__module__", "?"), e)
            return []

    # аргументы VK/MA — как в прежнем коде (не зависят от Яндекса)
    _vk_src = mood_query or (it.query or q)
    _vk_q = _mac_global._clean_query(_vk_src) or _vk_src
    _ma_src = mood_query or (search_q if effective_mix else (it.query or q))
    _want_parallel_pool = bool(effective_mix or mood_query)
    vk_task = None
    ma_task = None
    if source == "vk":
        vk_task = asyncio.ensure_future(_safe(_vkc_global.search, _vk_q, effective_limit))
    elif source == "auto" and _want_parallel_pool:
        vk_task = asyncio.ensure_future(_safe(_vkc_global.search, _vk_q, effective_limit))
    # MA параллельно — ТОЛЬКО для mix=1 (подборки): там нужен пул кандидатов.
    # Для mix=0 (точный поиск) MA НЕ запускаем — по транслиту приносит мусор
    # (Thiagão «Só Tua Graça Me Basta»), который вытесняет норм из топ-5.
    if source in ("auto", "yandex") and effective_mix:
        ma_task = asyncio.ensure_future(_safe(_mac_global.search, _ma_src, effective_limit))

    # ---- Yandex Music (source=auto|yandex) --------------------------------
    if source in ("auto", "yandex"):
        try:
            from . import ma_client as _mac
            from . import ym_client
            # «трек баста включи» → «баста»: тот же _clean_query, что и для MA.
            # mood_query (LLM по настроению) БЕРЁТСЯ ВМЕСТО исходного при mix.
            ym_src = mood_query or (it.query or q)
            ym_base = _mac._clean_query(ym_src) or ym_src
            # P1 ФИКС «джаз/электрон → глобальный мусор»: LLM-intent иногда даёт
            # ЖАНР НА ЛАТИНИЦЕ («jazz», «electronic»), а _query_variants для
            # латиницы не даёт русских якорей → глобальный шлак («Restaurant
            # Music Academy» вместо «Ночной Джаз»). Кириллица исходной просьбы
            # идёт ПЕРВОЙ (русский каталог Яндекса), латиница intent — как
            # дополнительный вариант. Если mood_query на кириллице — как есть.
            _cyr = re.search(r"[а-яё]", ym_base, re.I)
            _cyr_orig = re.search(r"[а-яё]", mood_base, re.I)
            if not _cyr and _cyr_orig:
                ym_q = _mac._clean_query(mood_base) or mood_base
            else:
                ym_q = ym_base
            # ДЕКАДНЫЙ ЯКОРЬ (ТЗ п.4/п.9): LLM-строка бывает падежной и
            # мусорной («классической русской рок 90-х» → Я даёт не то).
            # Строим чистый якорь из ИСХОДНОЙ просьбы: «русский рок 1990-х».
            # ВНИМАНИЕ: decade живёт в music_intent (MusicIntent), НЕ в it
            # (it — старый parse_intent из intent.py, у него нет decade).
            _dec_q = int(getattr(music_intent, "decade", 0) or 0) if music_intent else 0
            if _dec_q and (effective_mix or mood_query):
                _words = [
                    w for w in re.split(r"\W+", mood_base or q, flags=re.U)
                    if w and "классическ" not in w.lower().replace("ё", "е")
                ]
                _anchor = re.sub(r"\b\d{2}[- ]?(х|е)\b", "", " ".join(_words)).strip()
                if _anchor:
                    ym_q = f"{_anchor} {_dec_q}-х"
                    logger.info("decade anchor: %r (decade=%d)", ym_q, _dec_q)
            # P1: OVER-FETCH кандидатов. Раньше брали ровно limit (10) → после
            # age-фильтра/mood-фильтра оставалось 2-3 трека, рекомендатору
            # нечего ранжировать (журнал: 5 -> 3). Для подборок/mood берём
            # глубже (×3, кап 30), потом режем до effective_limit.
            # ФИКС (2026-10-31): mix=0 ТОЖЕ берём ×3 — Яндекс по подстроке
            # «баста» отдал Thiagão/Dirty Red (4 из 5 мусор), без over-fetch
            # нечего отфильтровать. Лимит Яндекса: fetch=min(limit*3, 30).
            _fetch_deep = min(effective_limit * 3, 30)
            from . import concurrency as _cc_sem
            async with _cc_sem.source_gate():
                ym_tracks = await ym_client.search(ym_q, limit=_fetch_deep)
            if ym_tracks:
                from .models import TrackMeta as _TM
                tracks = [_TM(**t) for t in ym_tracks]
                logger.info(
                    "search via Yandex: %d hits (fetch=%d) for %r",
                    len(tracks), _fetch_deep, ym_q,
                )
        except Exception as e:
            logger.warning("Yandex search failed: %s", e)

    # ---- VK Music (source=auto|vk) ------------------------------------------
    # ВК Музыка: api.vk.ru (основной, 2026-10-30+) или al_audio.php (fallback).
    # source=vk → ВК основной (Яндекс не вызываем).
    # source=auto → ВК добирает если Яндекс дал мало (точный поиск);
    # для подборок ВК УЖЕ запущен параллельно (vk_task выше).
    if source == "vk" or (source == "auto" and (vk_task is not None or len(tracks) < effective_limit)):
        try:
            from . import vk_client as _vkc
            from . import ma_client as _mac_v
            if vk_task is not None:
                _vk_tracks = await vk_task   # уже выполнялся параллельно с Яндексом
                vk_task = None
            else:
                _vk_src_seq = mood_query or (it.query or q)
                from . import concurrency as _cc_sem
                async with _cc_sem.source_gate():
                    _vk_tracks = await _vkc.search(
                        _mac_v._clean_query(_vk_src_seq) or _vk_src_seq,
                        limit=effective_limit - len(tracks) if source == "auto" else effective_limit,
                    )
            if _vk_tracks:
                from .models import TrackMeta as _TM
                # дедуп смысловых дублей (тот же трек может быть и в Яндексе)
                _have = {t.id for t in tracks}
                _have_keys = {
                    ((t.artist or "").lower().strip(), (t.title or "").lower().strip())
                    for t in tracks
                }
                added = 0
                for _vt in _vk_tracks:
                    if _vt["id"] in _have:
                        continue
                    _key = (_vt["artist"].lower().strip(), _vt["title"].lower().strip())
                    if _key in _have_keys:
                        continue
                    tracks.append(_TM(**_vt))
                    _have_keys.add(_key)
                    added += 1
                    if len(tracks) >= effective_limit:
                        break
                if added:
                    logger.info(
                        "search via VK Music: +%d hits for %r (total %d) [source=%s]",
                        added, _vk_src, len(tracks), source,
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("VK Music search failed: %s", e)

    # ---- Music Assistant (только source=auto|yandex, не для vk) ------------
    # YouTube-fallback ОТКЛЮЧЁН: при 0 от MA он уводил в yt-dlp →
    # «Sign in to confirm» → play не работал. Пусто = пусто (кэш ниже).
    # ФИКС (2026-10-31): MA НЕ вызывать если Яндекс уже дал ≥ effective_limit —
    # иначе MA по транслиту «basta» приносит Thiagão/Michael Basta (мусор),
    # которые в exact mode вытесняют нормальных из топ-5 (match@5 падает).
    _ma_needed = (
        source in ("auto", "yandex")
        and (
            ma_task is not None  # уже запущен параллельно (mix=1) — ждём
            or (len(tracks) < effective_limit and not effective_mix)  # mix=0: мало
            or (effective_mix and len(tracks) < effective_limit)      # mix=1: мало
        )
    )
    if _ma_needed:
        try:
            from . import ma_client
            ma_src = mood_query or (search_q if effective_mix else (it.query or q))
            if ma_task is not None:
                ma_tracks = await ma_task   # уже выполнялся параллельно с Яндексом
                ma_task = None
            else:
                from . import concurrency as _cc_sem
                async with _cc_sem.source_gate():
                    ma_tracks = await ma_client.search(
                        ma_src, limit=effective_limit - len(tracks)
                    )
            if ma_tracks:
                from .models import TrackMeta as _TM
                # дедуп: MA может принести то, что уже нашли Яндекс/ВК
                _have_keys_ma = {
                    ((t.artist or "").lower().strip(), (t.title or "").lower().strip())
                    for t in tracks
                }
                _added_ma = 0
                for _mt in ma_tracks:
                    _key = (_mt.get("artist", "").lower().strip(), _mt.get("title", "").lower().strip())
                    if _key in _have_keys_ma:
                        continue
                    tracks.append(_TM(**_mt))
                    _have_keys_ma.add(_key)
                    _added_ma += 1
                if _added_ma:
                    logger.info(
                        "search via Music Assistant: +%d hits for %r (total %d)",
                        _added_ma, ma_src, len(tracks),
                    )
            elif not tracks:
                logger.info("MA returned 0 hits for %r (no YouTube fallback)", ma_src)
        except Exception as e:
            logger.warning("MA search failed (no YouTube fallback): %s", e)
        finally:
            ma_task = None
    elif ma_task is not None:
        # Яндекс уже дал полный лимит — задача MA больше не нужна, отменяем
        ma_task.cancel()
        ma_task = None

    # Rate-limit / empty: serve a cached literal query so the user still sees
    # results instead of a blank screen (look-behind on the clean query).
    if not tracks and it.query:
        cached = search_cache.get(f"search:{it.query.lower()}:{min(limit, config.MAX_SEARCH_RESULTS)}")
        if cached:
            tracks = cached
            logger.info("search served from literal cache during fetch failure")

    # БЕНЧ 2026-10-31: ЗАПИСИ-РАДИО («DFM Дискач 90-х», «Дискотека 80-х &
    # 90-х», «Радио Рекорд Супердискотека») пришли из MA и встали в топ-1
    # на запросах «песни 80-х / музыка 90-х / хиты 2000-х» (match@5=0.00).
    # Настоящий трек ВСЕГДА имеет артиста; пустой artist = не-трек.
    # Фильтр по artist-строке фактически ВК не задевает (у ВК artist
    # заполнен даже у «Неизвестен»).
    #
    # + СПАМ-ARTIST (2026-10-31): хелпер `_artist_is_spam` — модульный
    # (см. выше, до @app.get("/search")); здесь и финальный проход перед
    # строгим лимитом ловит спам от top-up «{seed} хиты».
    _nj = [t for t in tracks
           if (t.artist or "").strip() and not _artist_is_spam(t.artist.strip())]
    if len(_nj) != len(tracks):
        logger.info("radio-junk /search: %d -> %d (без артиста/спам убраны)",
                    len(tracks), len(_nj))
        tracks = _nj

    # СТРОГИЙ ЛИМИТ выдачи (страховка): Яндекс/MA могут дать больше — режем
    # до effective_limit (поиск 10, mix 20), как просил пользователь.
    # ---- DECADE ENRICHMENT (ТЗ п.3.2 «похожие композиции/исполнители») ----
    # Эмпирика 2026-10-31: Яндекс по абстрактным «русский рок 90-х» отдаёт
    # СВЕЖИЕ подборки (проверено напрямую), артист-якорь даёт реальные 90-е
    # («ДДТ хиты» → ДДТ 1996/1998). Для decade-запросов второй проход:
    # ближайший к декаде ym-трек из кандидатов → его Rotor = похожее старьё.
    _dec_e = int(getattr(music_intent, "decade", 0) or 0) if music_intent else 0
    if _dec_e and effective_mix:
        from . import ym_client as _ymc_enr   # свой импорт: _ymc живёт ниже (blocklist)
        from .models import TrackMeta as _TM_dec
        _near, _best = None, 999
        for _t in tracks:
            _y = _t.year or 0
            if _y and str(_t.id).startswith("ym_"):
                _d = abs(_y - _dec_e)
                if _d < _best:
                    _best, _near = _d, _t
        if _near and 5 <= _best <= 25:
            try:
                _raw_rot, _b_rot = await _ymc_enr.rotor_tracks(
                    _near.id, limit=min(effective_limit * 2, 20)
                )
                _have = {t.id for t in tracks}
                _added = [t for t in _raw_rot if t.get("id") not in _have]
                tracks.extend(_TM_dec(**t) for t in _added)
                logger.info(
                    "decade enrichment: ротор от %r (%s, |%d-%d|=%d) -> +%d",
                    (_near.artist or "")[:24], _near.year, _near.year or 0, _dec_e,
                    _best, len(_added),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("decade enrichment failed: %s", e)

    # БЛОК-ЛИСТ мусора (Macarena и т.п.): Яндекс чистится в ym_client, но
    # Music Assistant (второй источник) приносит мусор мимо → чистим ЗДЕСЬ,
    # едино для ВСЕХ источников, ДО лимита (иначе мусор займёт слот).
    from . import ym_client as _ymc
    _before_blk = len(tracks)
    tracks = [t for t in tracks if not _ymc._is_blocked({"title": t.title, "artist": t.artist})]
    if len(tracks) != _before_blk:
        logger.info("blocklist /search: %d -> %d (убран мусор из MA/других)", _before_blk, len(tracks))

    # ФИЧА 4/1: дизлайки/скипы юзера — точный поиск режет только ДИЗЛАЙКИ
    # (скип «не сейчас» не мешает, если юзер сам ищет этот трек).
    from . import taste as _taste
    tracks = _taste.filter_tracks(tracks, hard_only=True)

    # ФИКС (2026-10-31): exact mode (mix=0) — ФИЛЬТР ПО АРТИСТУ-ЯКОРЮ.
    # Поиск «баста» → Яндекс дал 15 норм, MA добавил 5 мусора (Thiagão
    # «Só Tua Graça Me Basta», Michael Basta LCSW). Без фильтра мусор
    # содержит «Basta» как подстроку → artist_match=high → вытесняет норм
    # из топ-5 (live: 1 из 5 норм). Фильтр: ОСТАВЛЯЕМ треки, где артист
    # или title содержит токены якоря (баста↔basta, эминем↔eminem).
    # Если после фильтра < effective_limit — оставляем как есть (не рискуем
    # пустой выдачей: VK/MA могли дать релевантных, но без токенов).
    # ФИКС «исполнитель + название» («асти модный поп»): regex-intent видит
    # жанр («поп») → запрос уходит в ПОДБОРКУ (mode=mood) с мусорным якорем
    # («модный»), топ засран словом «Поп» по всему каталогу, а нужный
    # исполнитель 2-м (live: match@5=0.00). Фикс: если Яндекс реально нашёл
    # СИЛЬНЫЙ артист-хит (≥3 трека) и юзер НЕ просил набор ЯВНЫМ словом
    # (подборка/микс/хиты/...) — переходим в ТОЧНЫЙ режим, топ = исполнитель.
    # Жанр-подборки («веселая поп музыка») без артист-хитов не трогаем.
    if effective_mix:
        from . import recommender as _rec_x
        _q_low = (q or "").lower()
        _set_words = (
            "подборк", "микс", "плейлист", "радио", "список", "тренд",
            "лучш", "хит", "сборник", "что-нибудь", "что нибудь",
            "по настроен", "настроен", "похож", "друг",
        )
        if not any(w in _q_low for w in _set_words):
            _fix_seed, _fix_hits = _rec_x.best_seed_from_query(q, tracks)
            # Порог 1, не 3: выдача Яндекса нестабильна — «асти модный поп»
            # то даёт 5 хитов (08:51), то 1 (09:38) на одном запросе.
            # Достаточно ХОТЯ БЫ ОДНОГО артиста в пуле = запрос про артиста.
            # Слова-набора («подборка»/«хиты») уже отсечены выше.
            if _fix_hits >= 1:
                logger.info(
                    "search: якорь %r (%d артист-хитов) -> точный режим вместо подборки",
                    _fix_seed, _fix_hits,
                )
                effective_mix = 0
                mood_query = None

    if not effective_mix:
        from . import recommender as _rec  # свой импорт: _rec живёт ниже (line 562)
        # Якорь — ВЫБОР ПО РЕАЛЬНОМУ ПУЛУ («асти лампочки» → «асти», а не
        # «лампочки»; «баста осень» → «баста»). Слепая эвристика последнего
        # слова — только фоллбэк, когда пул не дал ни одного хита («кино»).
        _ca_seed, _ca_hits = _rec.best_seed_from_query(q, tracks)
        if not _ca_seed:
            _ca_seed, _ca_hits = _rec.best_seed_from_query(it.query or "", tracks)
        _seed_anchor = (
            _ca_seed
            or (music_intent.seed_artist if music_intent else "")
            or _rec._extract_artist_from_query(q)
            or _rec._extract_artist_from_query(it.query or "")
        )
        if _seed_anchor and len(_seed_anchor) >= 3:
            _a_norm = _rec._norm(_seed_anchor)
            _a_toks = set(_a_norm.split())  # токены якоря («баста» → {"basta"}; «Miyagi & Andy» → {"miyagi","andy"})
            if _a_toks:
                # ДВА УРОВНЯ (ТЗ: артист-якорь важнее title-матча):
                #   1) ARTIST-матч: треки, где артист содержит токен якоря
                #      («Баста» как исполнитель, «Кино» как группа) — СТРОГИЙ.
                #   2) TITLE-матч: если artist-матч дал <3, берём title тоже
                #      (Яндекс отдаёт «SLŸCH - Баста» — треки с названием якоря,
                #      но не артиста). Иначе «кино» = title всех треков «Кино»
                #      от NATCORN/ЛСП — мусор выше группы Кино.
                def _match_artist(t) -> bool:
                    _art = _rec._norm(t.artist or "")
                    return any(tok in _art for tok in _a_toks if len(tok) >= 3)

                def _match_title(t) -> bool:
                    _ttl = _rec._norm(t.title or "")
                    return any(tok in _ttl for tok in _a_toks if len(tok) >= 3)

                # СТРОГИЙ ARTIST-ХИТ (2026-10-31, жалоба «косяки по именам»):
                # подстроку seed ловили составные имена чужих народов —
                # «Dirty Red & Tha Basta», «The Ang Bandang Hindi Basta Basta»,
                # «Michael Basta, LCSW & Marcia J. Gomez, LCSW» → на «баста
                # без слим шейди» 4 из 5 хитов — мусор (бразильцы/филиппинцы).
                # Правило: artist-токены ⊇ seed-токены + НЕ БОЛЕЕ 1 лишнего
                # слова (служебные артистов не считаются).
                #   «Баста» → 0 ✓; «Linkin Park» → 1 ✓; «The Cranberries» →
                #   служебное the ✓; «Кино Фильм» → 1 ✓;
                #   «Dirty Red & Tha Basta» → 3 ✗; «Ang… Hindi Basta Basta»
                #   → 4 ✗; «Michael Basta, LCSW…» → 6 ✗.
                _SVC_TOKS = {
                    "the", "a", "an", "and", "of", "feat", "ft", "with", "x",
                    "vs", "or", "in", "on", "deb", "la", "le", "di", "de",
                }

                def _strong_name(artist_str: str) -> bool:
                    _arts = [w for w in _rec._norm(artist_str).split() if w.isalnum()]
                    _extra = [w for w in _arts
                              if w not in _a_toks and w not in _SVC_TOKS]
                    # ФИКС (2026-11-01) «Бриллиант пропал»: `SEVILLE, ARTIK & ASTI`
                    # = 2 лишних (seville, artik) → старый strict отбрасывал фит
                    # ASTI-трек. Теперь составное имя СИЛЬНОЕ, если seed-токен в нём
                    # есть (не больше 3 лишних = фит, а не чужой). «Джиган, ARTIK &
                    # ASTI» тоже сильный; «Só Tua Graça Me Basta» (нет seed) — слабый.
                    _seed_hit = bool(_a_toks) and (_a_toks & set(_arts))
                    return len(_extra) <= 1 or (_seed_hit and len(_extra) <= 3)

                def _strong_artist(t) -> bool:
                    return _strong_name(t.artist or "")

                _before_anchor = len(tracks)
                _artist_hits = [t for t in tracks if _match_artist(t)]
                _strong_hits = [t for t in _artist_hits if _strong_artist(t)]
                if _strong_hits and len(_strong_hits) < len(_artist_hits):
                    # Оставляем сильные, но ТОЛЬКО если хоть один есть
                    # (все слабые → не трогаем, иначе пустота).
                    logger.info(
                        "anchor strict-name /search: artist_hits %d -> %d "
                        "(составные имена-мусор убраны, seed=%r)",
                        len(_artist_hits), len(_strong_hits), _seed_anchor,
                    )
                    _artist_hits = _strong_hits
                _raw_anchor = list(tracks)  # сырой пул ДО якорной фильтрации
                if len(_artist_hits) >= 3 or _strong_hits:
                    # Достаточно артист-матчов → СТРОГИЙ: ВЫБРАСЫВАЕМ title-only
                    # (жалоба 2026-10-31 «Яндекс даёт дичь на Cranberries Zombie»:
                    # в топ-10 лезли «the sound chalk makes - cranberries»,
                    # «John Plata - Cranberries», «DRWZZLE» — треки с НАЗВАНИЕМ
                    # cranberries от чужих. ВК в том же запросе даёт сплошь
                    # The Cranberries → нужно так же. Пустота не страшна:
                    # top-up «{seed} хиты» доборит настоящими, а ранжер всё
                    # равно понизит title-only в хвост.
                    # «or _strong_hits» (2026-10-31): после строгого фильтра
                    # имён «баста без слим шейди» даёт artist_hits=1 (настоящая
                    # Баста) — без этого условия else-ветка подмешивала 5
                    # title-only мусоров (jaxxice/SLŸCH/Estraca «Баста»).
                    tracks = _artist_hits
                else:
                    # Мало артист-матчов («кино»: группа не в топе) → title-матч
                    # тоже допускаем, но помечаем artist-хиты выше (см. ниже)
                    tracks = [t for t in tracks if _match_artist(t) or _match_title(t)]
                # БЕНЧ 2026-10-31: якорь убил ВСЁ в 0 → пусто («последний хит
                # оливии родриго»: Яндекс нашёл 30 hits, но artist-матч
                # «Olivia Rodrigo» не совпал с кириллическим пулом → 0 →
                # «Ничего не нашла»). Пусто ХУЖЕ сырого топа: раз LLM дал seed,
                # а Яндекс искал по нему — доверяем его релевантности,
                # top-up «{seed} хиты» ниже доборит настоящих артистов.
                _anchor_fallback = False
                if not tracks and _raw_anchor:
                    tracks = _raw_anchor
                    _anchor_fallback = True
                logger.info(
                    "anchor filter /search: %d -> %d (seed=%r toks=%s artist_hits=%d)%s",
                    _before_anchor, len(tracks), _seed_anchor, _a_toks,
                    len(_artist_hits), " [FALLBACK raw]" if _anchor_fallback else "",
                )
                # ТЗ п.3/п.6: точный якорь с МАЛЫМ числом настоящих хитов
                # («баста без слим шейди» → artist_hits=2 из 15: остальное
                # title-мусор) → добор НАСТОЯЩИМИ треками «{seed} хиты».
                # 2026-10-31: второй шаблон «{seed} песни» — эмпирика:
                # «Баста хиты» по нынешнему Яндексу даёт 0 настоящих, а
                # «Баста песни» → 11 из 20 (тест 01 «включи басту» ≥5).
                if ((0 < len(_artist_hits) or _anchor_fallback)
                        and len(_artist_hits) < limit):
                    try:
                        from .models import TrackMeta as _TM_a
                        from . import ym_client as _ymc_tu  # ЛОКАЛЬНО (баг: global-импорт ниже → UnboundLocalError)
                        _have_a = {t.id for t in tracks}
                        for _tmpl in ("хиты", "песни"):
                            if len(_have_a) >= limit:
                                break
                            _extra = await _ymc_tu.search(
                                f"{_seed_anchor} {_tmpl}", limit=limit)
                            _added = [t for t in _extra
                                      if t["id"] not in _have_a
                                      and _ymc_tu.match_artist_anchor(t, _seed_anchor)
                                      # тот же строгий фильтр имён: «баста хиты»
                                      # иначе вернул бы Dirty Red/Michael обратно
                                      and _strong_name(str(t.get("artist") or ""))]
                            logger.info(
                                "anchor top-up '%s %s': +%d из %d (hits=%d, have=%d)",
                                _seed_anchor, _tmpl, len(_added), len(_extra),
                                len(_artist_hits), len(_have_a),
                            )
                            if _added:
                                tracks.extend(
                                    _TM_a(**t) for t in _added[: limit - len(tracks)])
                                _have_a.update(t["id"] for t in _added)
                    except Exception as _e_t:  # noqa: BLE001
                        logger.info("anchor top-up failed: %s", _e_t)

        # ФИКС «одно слово без артист-хитов» («кино»): Яндекс top-15 по одному
        # кириллическому слову НЕ отдаёт группу (только title-матчи чужих:
        # NATCORN/ЛСП). Дозапрашиваем «{q} группа» и берём ТОЛЬКО треки, где
        # артист содержит токен запроса: «кино группа» → КИНО ✓;
        # «лампочки группа» → чужие не проходят отбор по артисту (0 добавлено).
        if _ca_hits == 0 and len((q or "").split()) <= 2 and re.search(r"[а-яё]", (q or "").lower()):
            try:
                _extra_q = f"{(q or '').strip()} группа"
                _extra = await _ymc.search(_extra_q, limit=15)
            except Exception:  # noqa: BLE001
                _extra = []
            if _extra:
                from .models import TrackMeta as _TM_q
                _q_toks = set(_rec._norm(q).split())
                _have = {t.id for t in tracks}
                _added = 0
                for _t in _extra:
                    if str(_t.get("id") or "") in _have:
                        continue
                    _art_n = _rec._norm(str(_t.get("artist") or ""))
                    if any(x in _art_n for x in _q_toks if len(x) >= 3):
                        tracks.append(_TM_q(**_t))
                        _have.add(str(_t.get("id") or ""))
                        _added += 1
                if _added:
                    logger.info(
                        "search: дозапрос %r -> +%d треков с артистом %r",
                        _extra_q, _added, q,
                    )

    # ФИЧА 5: «другой/не этот исполнитель» → артист текущего трека в КОНЕЦ
    # выдачи (не выкидываем — пусть будет вариантом, но не топит).
    if ctx and re.search(r"друг\w*|не этого|не этот|не его", q, re.I):
        _cur_artist = ctx.split(" - ")[0].strip().lower()
        if _cur_artist:
            tracks = sorted(
                tracks,
                key=lambda t: _cur_artist in (t.artist or "").lower(),
            )

    # ---- RECOMMENDATION ENGINE (патч, feature-флаг USE_NEW_RANKER) ----------
    # СТАРЫЙ путь: tracks уже отранжированы внутри каждого источника (ym/vk
    # _rank_by_artist). Новый путь: единый ранжер со свежестью/разнообразием/
    # вкусом. Откат: USE_NEW_RANKER=0 → код рекомендателя не трогается.
    if config.USE_NEW_RANKER:
        try:
            from . import recommender as _rec
            from . import taste as _taste_rec
            _fb_rec = _taste_rec.stats() if hasattr(_taste_rec, "stats") else {}
            _mode_rec = "mood" if (effective_mix or mood_query) else "exact"
            # свежесть (ТЗ п.9): ГЛАВНЫЙ источник — freshness из MusicIntent
            # («классика/90-х» → old_ok даже в подборке; «поновее» → new/recent).
            # Дефолт прежний: точный без mood-запроса → не режем.
            if music_intent and music_intent.freshness in ("new", "recent", "any", "old_ok"):
                _fresh_rec = music_intent.freshness
            else:
                _fresh_rec = "old_ok" if not (effective_mix or mood_query) else "recent"
            # ТЗ п.5: discovery_level («давай что-нибудь новое») — поднимаем
            # свежесть, если сам запрос её не задал.
            if (music_intent and music_intent.discovery_level in ("medium", "high")
                    and _fresh_rec in ("any", "old_ok")):
                _fresh_rec = "new" if music_intent.discovery_level == "high" else "recent"
            # ТЗ п.5: diversity_preference → жёсткость лимита артиста
            # (high: больше разных артистов, low: можно повторяться).
            _mpa = config.RECOMMENDER_MAX_PER_ARTIST
            if music_intent and music_intent.diversity_preference:
                _mpa = {"high": 1, "medium": 2, "low": 3}.get(
                    music_intent.diversity_preference, config.RECOMMENDER_MAX_PER_ARTIST)
            _pool_seed, _pool_hits = _rec.best_seed_from_query(q, tracks)
            _seed_rec = (
                _pool_seed                            # выбор по РЕАЛЬНОМУ пулу (главный)
                or _rec._extract_artist_from_query(q)          # артист из просьбы (со стоп-фильтром)
                or _rec._extract_artist_from_query(
                    # FIX ТЗ п.13: music_intent может быть None (intent не собрали)
                    # → крашил ранжер «NoneType has no attribute seed_artist»
                    # (журнал 12:23/12:26/12:29 — подборки падали в фолбэк).
                    (music_intent.seed_artist if music_intent else "") or ""
                )
                or (ctx.split(" - ")[0] if ctx else "")     # текущий трек
            )
            # ФИКС: мусорный/служебный seed («дороги», «вечером», «послушать») →
            # не якорь. _extract_artist_from_query уже снял большинство, здесь
            # страховка на случай, если просочилось (слово <3 симв или из списка).
            if _seed_rec and (
                len(_seed_rec) < 3
                or _rec._norm(_seed_rec) in {
                    "дороги", "треки", "музыку", "музыка", "подборки",
                    "вечером", "послушать", "саунд", "утром", "ночью",
                    "работы", "тренировки", "дорога", "стола", "настроения",
                }
            ):
                _seed_rec = ""
            # Алиасы якоря из MusicBrainz (шаг 15 ТЗ п.8): «Miyagi» ↔ «Мияги».
            # Только когда есть якорь; кэш MB делает повторы бесплатными;
            # ошибка MB не должна ронять ранжирование.
            _aliases: set[str] = set()
            if _seed_rec and config.MB_ENABLED:
                try:
                    from . import mb_client as _mb
                    _al = await _mb.artist_aliases(_seed_rec)
                    _aliases = {_rec._norm(a) for a in _al if a and len(a) >= 3}
                except Exception as e:  # noqa: BLE001
                    logger.debug("mb aliases failed: %s", e)
            # CLAP-похожесть (ТЗ п.4/п.7) — ОТДЕЛЬНЫЙ механизм, кэш-only:
            # референс (по убыванию приоритета):
            #   1) audio→audio: текущий играющий трек ctx («артист - название») —
            #      ищем его id в кандидатах/кэше → эмбеддинг трека;
            #   2) text→audio: текст-эмбеддинг запроса (только если УЖЕ посчитан).
            # Инференс в запросе НЕ запускаем: треки/текст без эмбеддинга →
            # audio_sim=0, работают метаданные. Результат — лишь ОДИН признак
            # в rank(), итоговый порядок не определяет.
            _audio_sim: dict[str, float] = {}
            try:
                from . import embeddings as _emb
                _ref = None
                _tids = [t.id for t in tracks if getattr(t, "id", None)]
                # 1) audio→audio по текущему треку (ctx), если его эмбеддинг в кэше
                if ctx and _tids:
                    _cur_id = None
                    # ctx = «артист - название» → ищем id этого трека среди кандидатов
                    # (обычно текущий трек уже в кэше после первого прослушивания)
                    _cur_id = next((t.id for t in tracks if ctx.lower().split(" - ")[0] in (t.artist or "").lower()), None)
                    if _cur_id:
                        _ref = _emb.embedding_for(_cur_id, compute=False)
                # 2) text→audio: если текст уже посчитан
                if _ref is None:
                    _ref = _emb.get_text_embedding(mood_base or q, compute=False)
                if _ref and _tids:
                    _audio_sim = _emb.audio_similarities(_tids, _ref)
                    if _audio_sim:
                        logger.info("CLAP audio_sim (кэш-only): %d/%d кандидатов", len(_audio_sim), len(_tids))
            except Exception as e:  # noqa: BLE001
                logger.debug("clap audio_sim skipped: %s", e)
            tracks = _rec.rank(
                tracks,
                query=q,
                seed_artist=_seed_rec,
                mode=_mode_rec,
                freshness=_fresh_rec,
                genres=(music_intent.genres if music_intent else []),  # ТЗ 3.4: жанры из intent
                seed_aliases=_aliases,
                audio_sim=_audio_sim,
                banned_track_ids=_taste.banned_track_ids(hard_only=True),
                banned_artists=set(),  # точный поиск — только дизлайки, не баны артистов
                liked_artists={a.lower() for a in _taste.get_taste()},
                # ТЗ п.3: веса из истории (с затуханием) — градация вкуса
                artist_weights=_taste.artist_weights(),
                genre_weights=_taste.genre_weights(),
                # ТЗ п.5: «без этого исполнителя» из MusicIntent
                exclude_artists=(music_intent.exclude_artists
                                 if music_intent else None),
                # ТЗ п.6: допустимые отклонения от якоря (low/medium/high)
                similarity=(music_intent.similarity
                            if music_intent else "medium"),
                max_per_artist=_mpa,
            )
            logger.info(
                "recommender /search: mode=%s seed=%r -> %d | brief: sim=%s disc=%s div=%s excl=%d",
                _mode_rec, _seed_rec, len(tracks),
                (music_intent.similarity if music_intent else "-"),
                (music_intent.discovery_level if music_intent else "-"),
                (music_intent.diversity_preference if music_intent else "-"),
                len(music_intent.exclude_artists) if music_intent else 0,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("recommender failed, fallback: %s", e)
            try:
                from . import metrics as _mt_rf
                _mt_rf.event("search_rank_failed")
            except Exception:  # noqa: BLE001
                pass

    # ГРАДАЦИИ СВЕЖЕСТИ (ТЗ п.9) — ПОСЛЕ ранжера и ДО обрезки limit:
    # ранжер даёт качество, слои — правило запроса (ранжер при old_ok
    # поднимал бы свежее обратно наверх — исправлено переносом сюда).
    if effective_mix:
        def _fresh_group(t) -> int:
            y = t.year or 0
            if y >= 2024:
                return 0
            if y >= 2021:
                return 1
            if y == 0:
                return 2   # неизвестно → НЕ новое (ТЗ п.9)
            return 3       # классика — вниз, не удаляем (обычный режим)
        # Если юзер ПРОСИТ старое (freshness=old_ok из MusicIntent:
        # «классический рок 90-х») — порядок слоёв ИНВЕРТИРУЕМ: классика
        # наверх (релевантность запросу), свежее ниже, но НЕ удаляется.
        _wants_old = bool(music_intent and music_intent.freshness == "old_ok")
        if _wants_old:
            def _fresh_group(t) -> int:
                y = t.year or 0
                if y == 0:
                    return 1              # неизвестно — вторыми (не новое)
                if y < 2021:
                    return 0              # классика/старьё — ВВЕРХ
                return 2                   # актуальные/новинки — ниже
        _before_sorted = len(tracks)
        # Классика при old_ok: ближе к декаде запроса («90-х» → 1990) — выше.
        # ЖАНР-ЯКОРЬ (ТЗ п.5): трек без пересечения с жанрами intent
        # (попса из VK «Róisín Murphy» на запросе русского рока) — ВНИЗ,
        # даже если год близок к декаде.
        _dec = int(getattr(music_intent, "decade", 0) or 0) if music_intent else 0
        _g_anchor: set[str] = set()
        if music_intent and music_intent.genres:
            for _g in music_intent.genres:
                _g_anchor.update(w for w in _g.lower().replace("_", " ").split() if len(w) >= 4)
        if _wants_old and _dec:
            def _key_old(t) -> tuple:
                y = t.year or 0
                _tgs = " ".join(t.genres or []).lower()
                _gm = bool(_g_anchor) and any(w in _tgs for w in _g_anchor)
                if y and y < 2021:
                    return (0, 0, abs(y - _dec)) if _gm else (2, 0, abs(y - _dec))
                if y == 0:
                    return (1, 0, 0)
                return (3, 0, y)                      # свежее — в конец
            tracks = sorted(tracks, key=_key_old)
        else:
            tracks = sorted(tracks, key=_fresh_group)   # стабильная сортировка
        _old_kept = sum(1 for t in tracks if (t.year or 0) and t.year < 2021)
        if _old_kept:
            logger.info(
                "freshness tiers /search (mix): %d tracks reordered, %d classic %s (old_ok=%s)",
                _before_sorted, _old_kept,
                "ВВЕРХ" if _wants_old else "внизу", _wants_old,
            )

    # ФИНАЛЬНАЯ чистка спам-artists: top-up «{seed} хиты» и «{q} группа»
    # дозапрашиваются ПОСЛЕ первого radio-junk прохода — Яндекс снова
    # мог вернуть бред-artist («...Me Basta (2014) rappratodos...» →
    # в выдачу лезли бразильцы на «баста без слим шейди»).
    _nj_fin = [t for t in tracks if not _artist_is_spam((t.artist or "").strip())]
    if len(_nj_fin) != len(tracks):
        logger.info("radio-junk финал /search: %d -> %d (спам убран)",
                    len(tracks), len(_nj_fin))
        tracks = _nj_fin

    if len(tracks) > effective_limit:
        tracks = tracks[:effective_limit]

    # ---- Фраза для озвучки («как Алиса»): короткое объяснение, что включаем ─
    # LLM отвечает одной фразой + может вызвать play_music/next/stop (приложение
    # исполняет). При недоступности LLM — тактовая заглушка (не молчим).
    # P0 ШАБЛОНЫ: простые случаи («Нашёл N треков») — шаблон, БЕЗ LLM-вызова.
    from . import fastpath as _fp
    say: str | None = None
    if not tracks:
        say = _fp.say_template("empty")
    elif effective_mix:
        # подборка — шаблон (быстро, без LLM); LLM только если нужна живая фраза
        say = _fp.say_template("mix", len(tracks))
    else:
        say = _fp.say_template("found", len(tracks))
    # LLM переписывает say только если доступен (фраза станет живее) — не блокируем
    if say and _llm_say_enabled():
        try:
            context = (
                f"Играет: {ctx}. " if ctx else ""
            ) + (
                f"Подборка по запросу «{mood_query or mood_base}». "
                f"Найдено треков: {len(tracks)}."
                if effective_mix
                else f"Найдено треков: {len(tracks)}."
            )
            r = await llm.smart_response(q, context=context, timeout=2.5,
                                         role="fast")   # ТЗ п.11: say → FAST
            # П.9/п.15 ТЗ: LLM-say НЕ блокирует — шаблон уже готов;
            # если LLM не ответила за 2.5с (или упала) → остаёмся на шаблоне.
            _llm_say = _clean_say(r.get("text"))
            if _llm_say:
                say = _llm_say
        except Exception as e:  # noqa: BLE001
            logger.warning("smart_response failed (say template kept): %s", e)

    # ТЗ п.3: история ПОИСКА (запрос + метаданные результатов) — раньше не
    # писалась вообще. Питает «что послушать», контекст LLM и веса профиля.
    try:
        from . import history as _hist_s
        _hist_s.record_search(q, [
            (t.model_dump() if hasattr(t, "model_dump") else dict(t))
            for t in tracks[:50]
        ])
    except Exception as _he:  # noqa: BLE001
        logger.debug("history record_search: %s", _he)

    resp = SearchResponse(
        query=q,
        results=tracks,
        total=len(tracks),
        intent=it.intent,
        mix=bool(effective_mix),
        playlist=it.matched_playlist,
        say=say,
        mood_query=mood_query,
    )
    return resp


# ---------------------------------------------------------------------------
# GET /track/{video_id}
# ---------------------------------------------------------------------------
@app.get("/track/{video_id}", response_model=TrackMeta)
async def api_track(video_id: str):
    """Get metadata for a single track."""
    track = await get_track_metadata(video_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track {video_id} not found")
    return track


# ---------------------------------------------------------------------------
# GET|HEAD /stream/{video_id}  — the core proxy endpoint
# ---------------------------------------------------------------------------
# ---- Music Assistant (MA) — основной источник вместо заблокированного YouTube
# id с префиксом ma_ (или не 11 символов) => трек из MA, поток с 127.0.0.1:18097.
# ---------------------------------------------------------------------------
async def _ma_stream_response(video_id: str, is_head: bool, range_header: str | None):
    """Прокси потока Music Assistant (FLAC/mp3, session-id в URL).

    ВАЖНО: MA flow-стрим отдаётся НЕПРЕРЫВНО пока очередь играет (crossfade/
    flow mode) — полное чтение r.content никогда не завершится. Поэтому
    только streaming-прокси (StreamingResponse), без буферизации тела.
    """
    from . import ma_client

    try:
        url = await ma_client.ensure_stream(video_id)
    except Exception as e:
        logger.warning("MA ensure_stream failed for %s: %s", video_id, e)
        raise HTTPException(status_code=502, detail=f"MA stream: {e}")

    # ОТДЕЛЬНЫЙ клиент: общий get_http_client() использует local_address="::"
    # (принудительный IPv6 — нужен для YouTube CDN), а MA-поток живёт по
    # IPv4 (docker bridge 172.17.0.x) → «All connection attempts failed».
    client = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0, read=None))
    up_headers = {"Accept-Encoding": "identity"}
    if range_header:
        up_headers["Range"] = range_header

    if is_head:
        try:
            r = await client.head(url, headers=up_headers, timeout=10.0)
        finally:
            await client.aclose()
        return Response(status_code=200, headers={
            "Content-Type": r.headers.get("content-type", "audio/flac"),
            "Accept-Ranges": "bytes",
            "Content-Length": r.headers.get("content-length", "0"),
            "Cache-Control": "no-cache",
        })

    try:
        req = client.build_request("GET", url, headers=up_headers)
        up_resp = await client.send(req, stream=True)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        await client.aclose()
        logger.warning("MA upstream error: %s", e)
        raise HTTPException(status_code=502, detail=f"MA proxy: {e}")

    retried_404 = False  # локально: максимум ОДИН refresh на этот запрос
    if up_resp.status_code not in (200, 206):
        # 404/410 = сессия MA устарела — переиграть ОДИН раз. ВАЖНО: раньше
        # был бесконечный цикл (404 → pop → play (15с, URL ещё не активен) →
        # 404 → ExoPlayer retry → ...). Теперь: один refresh, дальше честный
        # 404 плееру (ExoPlayer сам обработает / переиграет следующий трек).
        if up_resp.status_code in (404, 410) and not retried_404:
            retried_404 = True
            await up_resp.aclose()
            ma_client._stream_cache.pop(video_id, None)
            try:
                url = await ma_client.ensure_stream(video_id)
                req = client.build_request("GET", url, headers=up_headers)
                up_resp = await client.send(req, stream=True)
            except Exception as retry_err:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"MA session refresh: {retry_err}")
        if up_resp.status_code not in (200, 206):
            status = up_resp.status_code
            await up_resp.aclose()
            await client.aclose()
            raise HTTPException(status_code=status, detail=f"MA returned {status}")

    resp_headers = {
        "Content-Type": up_resp.headers.get("content-type", "audio/flac"),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }
    for h in ("content-length", "content-range"):
        if h in up_resp.headers:
            resp_headers[
                "Content-Length" if h == "content-length" else "Content-Range"
            ] = up_resp.headers[h]

    async def _pipe():
        try:
            async for chunk in up_resp.aiter_bytes(chunk_size=65536):
                yield chunk
        finally:
            await up_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _pipe(),
        status_code=up_resp.status_code,
        headers=resp_headers,
    )


# ---------------------------------------------------------------------------
# ---- Yandex Music — свежий каталог (метаданные через прокси, файлы напрямую)
# id с префиксом ym_ => трек Яндекса, файл с storage.yandex.net (206 + Range,
# полноценная перемотка — в отличие от непрерывного MA-flow).
# ---------------------------------------------------------------------------
async def _ym_stream_response(video_id: str, is_head: bool, range_header: str | None):
    """Прокси файла Яндекс Музыки (mp3 192, Content-Length + Range)."""
    from . import ym_client

    try:
        url = await ym_client.cached_direct_link(video_id)
    except Exception as e:
        logger.warning("Yandex direct_link failed for %s: %s", video_id, e)
        raise HTTPException(status_code=502, detail=f"Yandex link: {e}")

    client = ym_client.stream_client()
    up_headers = {"Accept-Encoding": "identity"}
    if range_header:
        up_headers["Range"] = range_header

    if is_head:
        try:
            r = await client.head(url, headers=up_headers, timeout=10.0)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise HTTPException(status_code=502, detail=f"Yandex HEAD: {e}")
        return Response(
            status_code=200,
            headers={
                "Content-Type": r.headers.get("content-type", "audio/mpeg"),
                "Accept-Ranges": "bytes",
                "Content-Length": r.headers.get("content-length", "0"),
                "Cache-Control": "no-cache",
            },
        )

    async def _open(target_url: str):
        req = client.build_request("GET", target_url, headers=up_headers)
        return await client.send(req, stream=True)

    try:
        up_resp = await _open(url)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning("Yandex upstream error: %s", e)
        raise HTTPException(status_code=502, detail=f"Yandex proxy: {e}")

    # 410 = direct-link протух (живёт ~60с), 403 = право отозвали → ОДИН форс.
    if up_resp.status_code in (403, 410):
        await up_resp.aclose()
        try:
            url = await ym_client.cached_direct_link(video_id, force=True)
            up_resp = await _open(url)
        except Exception as retry_err:
            raise HTTPException(status_code=502, detail=f"Yandex refresh: {retry_err}")
        if up_resp.status_code not in (200, 206):
            status = up_resp.status_code
            await up_resp.aclose()
            raise HTTPException(status_code=status, detail=f"Yandex returned {status}")

    if up_resp.status_code not in (200, 206):
        status = up_resp.status_code
        await up_resp.aclose()
        raise HTTPException(status_code=status, detail=f"Yandex returned {status}")

    resp_headers = {
        "Content-Type": up_resp.headers.get("content-type", "audio/mpeg"),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }
    for h in ("content-length", "content-range"):
        if h in up_resp.headers:
            resp_headers[
                "Content-Length" if h == "content-length" else "Content-Range"
            ] = up_resp.headers[h]

    async def _pipe():
        try:
            async for chunk in up_resp.aiter_bytes(chunk_size=65536):
                yield chunk
        finally:
            await up_resp.aclose()

    return StreamingResponse(
        _pipe(),
        status_code=up_resp.status_code,
        headers=resp_headers,
    )


async def _vk_stream_response(video_id: str, is_head: bool, range_header: str | None):
    """Прокси файла VK Музыки (mp3/aac с vkuseraudio CDN, Range → 206)."""
    from . import vk_client

    try:
        url = await vk_client.resolve_stream(video_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("VK resolve failed for %s: %s", video_id, e)
        raise HTTPException(status_code=502, detail=f"VK resolve: {e}")

    client = vk_client.stream_client()
    up_headers = {"Accept-Encoding": "identity"}
    if range_header:
        up_headers["Range"] = range_header

    if is_head:
        try:
            r = await client.head(url, headers=up_headers, timeout=10.0)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise HTTPException(status_code=502, detail=f"VK HEAD: {e}")
        return Response(
            status_code=200,
            headers={
                "Content-Type": r.headers.get("content-type", "audio/mpeg"),
                "Accept-Ranges": "bytes",
                "Content-Length": r.headers.get("content-length", "0"),
                "Cache-Control": "no-cache",
            },
        )

    async def _open(target_url: str):
        req = client.build_request("GET", target_url, headers=up_headers)
        return await client.send(req, stream=True)

    try:
        up_resp = await _open(url)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning("VK upstream error: %s", e)
        raise HTTPException(status_code=502, detail=f"VK proxy: {e}")

    # 403/410 = CDN-ссылка протухла (expires ~сутки) → ОДИН форс-рефреш reload.
    if up_resp.status_code in (403, 410):
        await up_resp.aclose()
        vk_client._stream_cache.pop(video_id, None)
        try:
            url = await vk_client.resolve_stream(video_id)
            up_resp = await _open(url)
        except Exception as retry_err:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"VK refresh: {retry_err}")
        if up_resp.status_code not in (200, 206):
            status = up_resp.status_code
            await up_resp.aclose()
            raise HTTPException(status_code=status, detail=f"VK returned {status}")

    if up_resp.status_code not in (200, 206):
        status = up_resp.status_code
        await up_resp.aclose()
        raise HTTPException(status_code=status, detail=f"VK returned {status}")

    resp_headers = {
        "Content-Type": up_resp.headers.get("content-type", "audio/mpeg"),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }
    for h in ("content-length", "content-range"):
        if h in up_resp.headers:
            resp_headers["Content-Length" if h == "content-length" else "Content-Range"] = up_resp.headers[h]

    async def _pipe():
        try:
            async for chunk in up_resp.aiter_bytes(chunk_size=65536):
                yield chunk
        finally:
            await up_resp.aclose()

    return StreamingResponse(
        _pipe(),
        status_code=up_resp.status_code,
        headers=resp_headers,
    )


async def _resolve_with_retry(video_id: str):
    """Resolve stream URL; retry once on transient failures.

    Deliberately does NOT retry BotBlocked — that's a YouTube rate-limit;
    retrying just adds another player request to a hot IP. The route answers
    503 + Retry-After immediately instead of hanging the client.
    """
    try:
        return await resolve_stream_url(video_id)
    except BotBlocked:
        raise
    except RuntimeError as first_err:
        logger.warning("resolve failed for %s (%s); retrying", video_id, first_err)
        stream_cache.delete(f"stream:{video_id}")
        await asyncio.sleep(1.0)
        return await resolve_stream_url(video_id)


@app.api_route("/stream/{video_id}", methods=["GET", "HEAD"])
async def api_stream(
    video_id: str,
    request: Request,
    range: Optional[str] = Header(None, alias="Range"),
):
    """Proxy audio stream from YouTube CDN.

    Supports:
        - HEAD requests (ExoPlayer probes with HEAD first)
        - HTTP Range requests (206 Partial Content) for seeking/buffering
        - Accept-Ranges: bytes header (CRITICAL for ExoPlayer)

    ExoPlayer flow:
        1. HEAD /stream/{id}  → gets Content-Length, Accept-Ranges
        2. GET /stream/{id}   → initial probe
        3. GET /stream/{id} Range: bytes=X-Y  → seek/buffer
    """
    # Yandex-трек (свежий каталог)? -> файл storage.yandex.net напрямую
    # (206 + Range + Content-Length → перемотка работает).
    # ВАЖНО: ПЕРЕД ma_ веткой — иначе is_ma_id ловит ym_* как «не 11 символов».
    from . import ym_client as _ym
    if _ym.is_ym_id(video_id):
        return await _ym_stream_response(video_id, is_head=(request.method == "HEAD"), range_header=range)

    # VK Music-трек (третий источник)? -> проксируем CDN vkuseraudio (mp3/aac)
    from . import vk_client as _vk
    if _vk.is_vk_id(video_id):
        return await _vk_stream_response(video_id, is_head=(request.method == "HEAD"), range_header=range)

    # MA-трек (новый источник)? -> прокси MA-потока, YouTube не трогаем
    from . import ma_client as _ma
    if _ma.is_ma_id(video_id):
        return await _ma_stream_response(video_id, is_head=(request.method == "HEAD"), range_header=range)

    # HEAD request — just return headers, no body
    if request.method == "HEAD":
        try:
            stream_info = await _resolve_with_retry(video_id)
        except BotBlocked:
            raise HTTPException(
                status_code=503,
                detail="YouTube rate-limit, retry shortly",
                headers={"Retry-After": "30"},
            )
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

        # Get upstream content-length via HEAD
        upstream_headers = dict(stream_info.http_headers)
        upstream_headers["User-Agent"] = upstream_headers.get(
            "User-Agent",
            "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/125.0"
        )
        upstream_headers["Accept-Encoding"] = "identity"

        client = await get_http_client()
        try:
            head_resp = await client.head(
                stream_info.direct_url,
                headers=upstream_headers,
                timeout=10.0,
            )
            total_size = int(head_resp.headers.get("content-length", 0))
            content_type = head_resp.headers.get("content-type", "audio/webm")
        except Exception:
            # If HEAD fails, return minimal headers
            total_size = 0
            content_type = "audio/webm"

        resp_headers = {
            "Content-Type": content_type,
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        }
        if total_size > 0:
            resp_headers["Content-Length"] = str(total_size)

        return Response(status_code=200, headers=resp_headers)

    # GET request — proxy the audio stream
    try:
        stream_info = await _resolve_with_retry(video_id)
    except BotBlocked:
        # Fast answer instead of hanging: phone retries on the next play.
        raise HTTPException(
            status_code=503,
            detail="YouTube rate-limit, retry shortly",
            headers={"Retry-After": "30"},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Build upstream request headers
    upstream_headers = dict(stream_info.http_headers)
    upstream_headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/125.0"
    )
    # googlevideo returns 403 when the client asks for gzip/deflate.
    # Audio is already compressed — force identity encoding.
    upstream_headers["Accept-Encoding"] = "identity"

    # Forward Range header if present (critical for ExoPlayer seeking)
    if range:
        upstream_headers["Range"] = range

    client = await get_http_client()

    # Fetch upstream content-length (needed for Content-Length in response)
    total_size = 0
    try:
        head_resp = await client.head(
            stream_info.direct_url,
            headers=upstream_headers,
            timeout=10.0,
        )
        total_size = int(head_resp.headers.get("content-length", 0))
    except Exception:
        pass  # Will proceed without total_size on GET

    # Proxy the actual request
    try:
        upstream_resp = await client.get(
            stream_info.direct_url,
            headers=upstream_headers,
            timeout=config.PROXY_TIMEOUT,
        )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        # URL may have expired — clear cache and retry once
        stream_cache.delete(f"stream:{video_id}")
        try:
            stream_info = await _resolve_with_retry(video_id)
            upstream_resp = await client.get(
                stream_info.direct_url,
                headers=upstream_headers,
                timeout=config.PROXY_TIMEOUT,
            )
        except Exception as retry_err:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch stream: {retry_err}",
            )

    if upstream_resp.status_code not in (200, 206):
        stream_cache.delete(f"stream:{video_id}")
        raise HTTPException(
            status_code=upstream_resp.status_code,
            detail=f"YouTube returned {upstream_resp.status_code}",
        )

    # Determine content type
    content_type = upstream_resp.headers.get("content-type", "audio/webm")

    # Upstream Content-Length
    upstream_cl = int(upstream_resp.headers.get("content-length", 0))

    # Use upstream CL if we don't have total_size from HEAD
    if total_size == 0 and upstream_cl > 0:
        total_size = upstream_cl

    # Build response headers — ALL critical for ExoPlayer
    resp_headers = {
        "Content-Type": content_type,
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }

    status_code = upstream_resp.status_code  # 200 or 206

    if status_code == 206:
        # Partial Content — forward Content-Range
        content_range = upstream_resp.headers.get("content-range")
        if content_range:
            resp_headers["Content-Range"] = content_range
            # Content-Length = end - start + 1 from Content-Range
            resp_headers["Content-Length"] = str(upstream_cl)
        elif total_size > 0 and range:
            # Build Content-Range manually
            start, end = parse_range_header(range, total_size)
            resp_headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
            resp_headers["Content-Length"] = str(end - start + 1)
        else:
            resp_headers["Content-Length"] = str(upstream_cl)
    else:
        # Full content
        if total_size > 0:
            resp_headers["Content-Length"] = str(total_size)
        elif upstream_cl > 0:
            resp_headers["Content-Length"] = str(upstream_cl)

    async def stream_generator():
        """Yield response chunks — no full-file buffering."""
        try:
            async for chunk in upstream_resp.aiter_bytes(
                chunk_size=config.STREAM_CHUNK_SIZE
            ):
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        stream_generator(),
        status_code=status_code,
        headers=resp_headers,
        media_type=content_type,
    )


# ---------------------------------------------------------------------------
# Playlists API
# ---------------------------------------------------------------------------
class PlaylistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class PlaylistTrackAdd(BaseModel):
    id: str
    title: str
    artist: str = ""
    duration: int = 0
    thumbnail: str = ""


@app.get("/api/playlists")
async def api_list_playlists():
    return {"playlists": playlist_store.list_playlists()}


@app.post("/api/playlists", status_code=201)
async def api_create_playlist(body: PlaylistCreate):
    return playlist_store.create_playlist(body.name)


@app.get("/api/playlists/{playlist_id}")
async def api_get_playlist(playlist_id: str):
    p = playlist_store.get_playlist(playlist_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return p


@app.delete("/api/playlists/{playlist_id}", status_code=204)
async def api_delete_playlist(playlist_id: str):
    if not playlist_store.delete_playlist(playlist_id):
        raise HTTPException(status_code=404, detail="Playlist not found")
    return Response(status_code=204)


@app.post("/api/playlists/{playlist_id}/tracks", status_code=201)
async def api_add_track(playlist_id: str, body: PlaylistTrackAdd):
    p = playlist_store.add_track(playlist_id, body.model_dump())
    if p is None:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return p


@app.delete("/api/playlists/{playlist_id}/tracks/{track_id}", status_code=204)
async def api_remove_track(playlist_id: str, track_id: str):
    p = playlist_store.remove_track(playlist_id, track_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Playlist not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def index():
    index_path = Path(config.STATIC_DIR) / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cache": {
            "search": len(search_cache),
            "stream": len(stream_cache),
            "metadata": len(metadata_cache),
        },
    }


# ---------------------------------------------------------------------------
# GET /speak?text=...   — озвучить фразу (TTS, русский голос edge-tts) → mp3.
# Используется приложением для ОЗВУЧКИ ответов ассистента («как Алиса»).
# Возвращает audio/mpeg; при недоступности TTS — 503 (клиент игнорирует).
# ---------------------------------------------------------------------------
@app.get("/speak")
async def api_speak(text: str = Query("", min_length=1, max_length=500)):
    mp3 = await llm.tts_mp3(text)
    if not mp3:
        raise HTTPException(status_code=503, detail="TTS unavailable")
    return Response(
        content=mp3,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(mp3)),
            "Cache-Control": "public, max-age=3600",
        },
    )


# ---------------------------------------------------------------------------
# РЕЖИМЫ РАДИО (ТЗ п.12, патч 2): mode=mix | discover | mood | similar.
# Все четыре используют общий Recommendation Engine (режимы намеренно
# отличаются СБОРОМ кандидатов, а не отдельными движками).
#   similar  — прежний «похожее на текущий» (дефолт, код ниже НЕ тронут)
#   mix      — «Мой микс»: личная станция Rotor от вкуса + добор по вкусу
#   discover — «Открытия»: только НОВЫЕ артисты (вне known_artists())
#   mood     — «По настроению»: mood-запрос, без привязки к текущему треку
# Возвращает SearchResponse или None (= ветвиться в прежний код similar).
# ---------------------------------------------------------------------------
async def _radio_mode(
    mode: str, *, id: str, limit: int, note: str, ctx: str, queue: str, fresh: int
):
    import random

    from . import fastpath as _fp
    from . import llm as _llm
    from . import recommender as _rec_engine
    from . import taste as _taste
    from . import ym_client
    from .models import TrackMeta as _TM

    if mode == "similar":
        return None

    tracks: list = []
    q_note = note or ""
    # ТЗ п.7: реакции «слишком однообразно / не повторяй» приходят note-ом
    # от /smart (см. quality-reaction) → жёсткое разнообразие в ранжере.
    _div_high = bool(re.search(
        r"однообраз|разнообраз|не повтор|повторяе|новых артист|други[ех] артист|"
        r"без повтор", f"{note} {ctx}".lower()))

    def _clap_sim(tracks_list: list) -> dict[str, float]:
        """CLAP audio_sim для radio (ТЗ п.4.4): кэш-only, референс — seed-трек
        (id). Нет эмбеддинга у seed/кандидатов → {} (работают метаданные);
        инференс в запросе НЕ запускается, модель не грузится."""
        try:
            if not (id and ym_client.is_ym_id(id)):
                return {}
            from . import embeddings as _emb
            _ref = _emb.embedding_for(id, compute=False)
            if not _ref:
                return {}
            _tids = [t.id for t in tracks_list if getattr(t, "id", None)]
            _sims = _emb.audio_similarities(_tids, _ref)
            if _sims:
                logger.info(
                    "CLAP audio_sim radio (кэш-only): %d/%d кандидатов", len(_sims), len(_tids)
                )
            return _sims
        except Exception as e:  # noqa: BLE001
            logger.debug("radio clap skipped: %s", e)
            return {}

    if mode == "mix":
        # «Мой микс» (ТЗ 12.1): станция Rotor от трека ЛЮБИМОГО артиста.
        # ФИКС: станция берётся от ТОЧНОГО совпадения вкусового артиста
        # (иначе «баста» → первый попавшийся чужой Basta → чужая станция).
        import re as _re
        liked = [
            a for a in (_taste.get_taste() or [])
            if 2 <= len(a) <= 40 and not _re.search(r"(.)\1{4,}", a)   # без склеек визарда
        ]
        station = id
        if not ym_client.is_ym_id(station):
            for a in liked[:5]:
                try:
                    found = await ym_client.search(a, limit=5)
                except Exception:  # noqa: BLE001
                    continue
                exact = [
                    t for t in found
                    if ym_client.match_artist_anchor(t, a)
                ]
                cand = (exact or found)[0] if (exact or found) else None
                if cand and str(cand.get("id", "")).startswith("ym_"):
                    station = cand["id"]
                    break
        if ym_client.is_ym_id(station):
            try:
                raw, _b = await ym_client.rotor_tracks(
                    station,
                    limit=limit,
                    queue_id=queue if (fresh and ym_client.is_ym_id(queue)) else None,
                )
                tracks = [_TM(**t) for t in raw]
            except Exception as e:  # noqa: BLE001
                logger.warning("mix rotor failed: %s", e)
        # добор: хиты артистов вкуса (свежие+популярные), когда ротора мало.
        # ФИКС: строгий отбор по артисту — иначе «баста хиты» приносит
        # title-мусор («SLŸCH - Баста» от чужого исполнителя).
        if len(tracks) < limit and liked:
            for a in liked[:3]:
                try:
                    extra = await ym_client.search(f"{a} хиты", limit=limit)
                except Exception:  # noqa: BLE001
                    continue
                have = {t.id for t in tracks}
                tracks.extend(
                    _TM(**t) for t in extra
                    if t["id"] not in have and ym_client.match_artist_anchor(t, a)
                )
                if len(tracks) >= limit:
                    break
        # ФИКС «title-мусор микса» (жалоба «одна Баста накидала»): трек, где
        # имя вкусового артиста ТОЛЬКО в названии («SIBERIAN - Баста»), а в
        # самом артисте НЕТ — мусор. Убираем, когда у якоря ≥3 настоящих
        # артист-хитов (иначе щадим — мало данных, не рискуем пустотой).
        if liked:
            _l_toks: set = set()
            for _la in liked:
                _l_toks |= set(_rec_engine._norm(_la).split())
            _l_toks = {x for x in _l_toks if len(x) >= 3}
            if _l_toks:
                def _has_in_arts(t) -> bool:
                    _an = _rec_engine._norm(getattr(t, "artist", "") or "")
                    return any(x in _an for x in _l_toks)
                _true_hits = sum(1 for t in tracks if _has_in_arts(t))
                # Порог 1: ХОТЯ Б ОДИН настоящий артист подтверждает якорь
                # (на его миксе была 1 настоящая Баста + 5 title-мусора —
                # при пороге 3 фильтр бы не отработал).
                if _true_hits >= 1:
                    _before_tm = len(tracks)
                    tracks = [
                        t for t in tracks
                        if _has_in_arts(t) or not any(
                            x in _rec_engine._norm(getattr(t, "title", "") or "")
                            for x in _l_toks
                        )
                    ]
                    if len(tracks) != _before_tm:
                        logger.info(
                            "mix title-musor: %d -> %d (убраны title-only, якоря %s)",
                            _before_tm, len(tracks), sorted(_l_toks)[:3],
                        )
        # ФИЛЬТР детского (реальный жанр Яндекса ИЛИ имя артиста) — для
        # подборки взрослого пользователя детское неуместно (локально для mix).
        tracks = [
            t for t in tracks
            if not (
                any(g in {"детский", "children", "kids"} for g in (t.genres or []))
                or "детск" in (t.artist or "").lower()
            )
        ]
        # Ранжер: бонус артистам вкуса + свежесть, снятие повторов (общий движок)
        # + CLAP audio_sim (кэш-only) — «Микс» тоже использует аудио-похожесть
        try:
            tracks = _rec_engine.rank(
                tracks,
                query="",
                mode="artist_radio",
                freshness="old_ok",
                liked_artists={a.lower() for a in liked},
                # ТЗ п.3: веса из истории — «мой микс» учитывает накопленный вкус
                artist_weights=_taste.artist_weights(),
                genre_weights=_taste.genre_weights(),
                audio_sim=_clap_sim(tracks),
                # ТЗ п.7: «слишком однообразно» → 1 трек на артиста
                max_per_artist=1 if _div_high else 2,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("mix rank failed: %s", e)

    elif mode == "discover":
        # «Открытия» (ТЗ 12.3): кандидаты MINUS уже знакомые артисты.
        known = _taste.known_artists()
        banned_ids = _taste.banned_track_ids(hard_only=True)

        async def _sim_c():
            return await ym_client.similar_tracks(id, limit=40) if ym_client.is_ym_id(id) else []

        async def _sea_c():
            _base = q_note or ym_client._query_genre(f"{note} {ctx}".strip()) or "новинки"
            try:
                return await ym_client.search(_base, limit=30)
            except Exception:  # noqa: BLE001
                return []

        res = await asyncio.gather(_sim_c(), _sea_c(), return_exceptions=True)
        cands: list[dict] = []
        for r in res:
            if isinstance(r, list):
                cands.extend(r)
        # фильтры: баны треков + знакомые артисты (сравнение по токенам)
        fresh_pool: list = []
        seen_ids: set[str] = set()
        seen_artists: set[str] = set()
        for c in cands:
            cid = str(c.get("id") or "")
            if not cid or cid in seen_ids or cid in banned_ids:
                continue
            seen_ids.add(cid)
            a_toks = ym_client._norm_artist(str(c.get("artist") or "")).split()
            # знаком? любой токен артиста ∈ known → пропускаем
            if known and any(any(k in tok or tok in k for k in known if len(k) >= 4) for tok in a_toks if len(tok) >= 4):
                continue
            if a_toks:
                _akey = " ".join(sorted(a_toks))
                if _akey in seen_artists:
                    continue   # максимум 1 трек на нового артиста
                seen_artists.add(_akey)
            fresh_pool.append(_TM(**c))
        random.shuffle(fresh_pool)   # разнообразие понятно → случайный порядок
        tracks = fresh_pool[:limit]
        # страховка: новых мало → не пустим пусто, добираем без артист-фильтра
        if len(tracks) < 3:
            for c in cands:
                if len(tracks) >= 3:
                    break
                if str(c.get("id")) not in {t.id for t in tracks} and str(c.get("id")) not in banned_ids:
                    tracks.append(_TM(**c))
        # Единый ранжер + CLAP (ТЗ п.12 / «Свежий»): веса mode="discovery"
        # (свежесть 35 / вкус 30 / audio_sim 20) вместо голого shuffle —
        # порядок решают метаданные + аудио-похожесть к seed (кэш-only).
        # Ошибка ранжера не роняет выдачу (shuffle-порядок остаётся).
        try:
            tracks = _rec_engine.rank(
                tracks,
                query=q_note,
                mode="discovery",
                freshness="recent",
                max_per_artist=2,
                audio_sim=_clap_sim(tracks),
                banned_track_ids=banned_ids,
                liked_artists=set(),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("discover rank failed: %s", e)

    elif mode == "mood":
        # «По настроению» (ТЗ 12.4): mood-запрос без привязки к текущему треку.
        base = q_note or "подборка"
        mood_query = None
        try:
            mood_query = await _llm.mood_mix_query(base, current_track=ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("mood mode llm failed: %s", e)
        mood_query = mood_query or f"{base} подборка"
        try:
            raw = await ym_client.search(mood_query, limit=min(limit * 3, 30))
            tracks = [_TM(**t) for t in raw]
        except Exception as e:  # noqa: BLE001
            logger.warning("mood mode search failed: %s", e)
        # градации свежести (ТЗ п.9): свежее выше, классика вниз, не удаляем
        def _fg(t) -> int:
            y = t.year or 0
            if y >= 2024:
                return 0
            if y >= 2021:
                return 1
            if y == 0:
                return 2
            return 3
        tracks = sorted(tracks, key=_fg)
        # общий движок: снятие дублей артиста + штраф недавних повторов
        # + CLAP audio_sim (кэш-only) — «Настроение» тоже использует аудио-похожесть
        try:
            tracks = _rec_engine.rank(
                tracks,
                query=q_note,
                mode="mood",
                freshness="recent",
                # ТЗ п.7: реакция «однообразно» → максимум разных артистов
                max_per_artist=1 if _div_high else 2,
                audio_sim=_clap_sim(tracks),
                banned_track_ids=_taste.banned_track_ids(hard_only=True),
                # ТЗ п.3: веса вкуса из истории (градация вместо «0.5 всем»)
                artist_weights=_taste.artist_weights(),
                genre_weights=_taste.genre_weights(),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("mood mode rank failed: %s", e)

    else:
        return None

    tracks = tracks[:limit]
    from . import taste as _taste2
    tracks = _taste2.filter_tracks(tracks, hard_only=True)
    # ТЗ п.3: выдача радио попадает в историю (metadанa для «что послушать»)
    try:
        from . import history as _hist_r
        _hist_r.record_tracks_meta([
            (t.model_dump() if hasattr(t, "model_dump") else dict(t))
            for t in tracks
        ])
    except Exception:  # noqa: BLE001
        pass
    say = _fp.say_template("mix", len(tracks)) if tracks else _fp.say_template("empty")
    return SearchResponse(
        query=f"radio:{mode}",
        results=tracks,
        total=len(tracks),
        intent="playlist",
        mix=True,
        say=say,
        mood_query=q_note or None,
    )


# ---------------------------------------------------------------------------
# GET /radio?id=ym_...  — РЕКОМЕНДАТЕЛЬНЫЙ ПУЛ: похожие треки на текущий.
# Источник НЕЗАВИСИМ от выдачи поиска (движок Яндекса /tracks/{id}/similar).
# Из него берутся «следующие» треки — как «Моя волна», а не из списка поиска.
# ---------------------------------------------------------------------------
@app.get("/radio", response_model=SearchResponse)
async def api_radio(
    id: str = Query("", max_length=64),        # текущий трек ym_<id>
    limit: int = Query(30, ge=1, le=60),
    note: str = Query("", max_length=200),     # подсказка от LLM («более спокойное»)
    ctx: str = Query("", max_length=300),      # «артист - название» для фразы
    queue: str = Query("", max_length=64),     # id последнего трека сессии → СВЕЖИЙ батч
    fresh: int = Query(0),                     # 1 = только Rotor (разные треки, не из списка)
    # ТЗ п.12: 4 режима радио. similar (дефолт) = прежний код без изменений.
    mode: str = Query("similar", max_length=16),  # similar|mix|discover|mood
):
    from . import ym_client

    # ТЗ «Похожие» 2026-10-31: v2 (параллельный сбор + CLAP + единый rank + MMR)
    # обрабатывается ОТДЕЛЬНО; исключение или None → прежний код ниже
    # (фолбэк, п.10 регрессия). Откат: USE_SIMILAR_V2=0.
    if mode == "similar" and config.USE_SIMILAR_V2:
        try:
            from .similar import build_similar as _similar_v2
            resp = await _similar_v2(
                id=id, limit=limit, note=note, ctx=ctx, queue=queue, fresh=fresh
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("similar v2 failed, fallback old: %s", e)
            resp = None
        if resp is not None:
            logger.info("radio mode=similar v2 -> %d tracks", len(resp.results))
            return resp

    # Патч 2 (ТЗ п.12): режимы mix/discover/mood обрабатываются ОТДЕЛЬНО;
    # None = «similar» → прежний код ниже (обратная совместимость 100%).
    if mode in ("mix", "discover", "mood"):
        try:
            resp = await _radio_mode(
                mode, id=id, limit=limit, note=note, ctx=ctx, queue=queue, fresh=fresh
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("radio mode %s failed, fallback similar: %s", mode, e)
            resp = None
        if resp is not None:
            logger.info("radio mode=%s -> %d tracks", mode, len(resp.results))
            return resp

    tracks: list = []
    batch_id: str | None = None
    # ФИКС скоупа: _genre_anchor объявляем ЗДЕСЬ (до if is_ym_id), иначе при
    # пустом id ниже (elif/mood) — NameError. Жанр-якорь из note+ctx.
    _genre_anchor = ym_client._query_genre(f"{note} {ctx}".strip())
    # АРТИСТ-ЯКОРЬ (жалоба 2026-10-30: ротор от «Баста-Выпускной» → Danza
    # Kuduro/Авария/Crazy Frog/Queen): артист из ctx (формат «artist - title»)
    # ИЛИ одиночное имя в note («баста»). Если есть артист ИЛИ жанр — ротор
    # фильтруется по якорю, чужое/старое отбрасывается, ниже mood-добор.
    _artist_anchor = ym_client._artist_anchor_from_ctx(ctx) or ym_client._artist_anchor_from_ctx(note)
    # Есть ли хоть какой-то якорь (жанр или артист) — тогда ротор не пускаем вслепую.
    _has_anchor = bool(_genre_anchor or _artist_anchor)
    if ym_client.is_ym_id(id):
        # 1) Rotor-сессия — личный поток Яндекса (как «Моя волна»), приоритет.
        #    ПРОВЕРЕНО живьём: одна станция track:X с разными queue возвращает
        #    ПОЧТИ тот же топ (8 уник./20), а СМЕНА станции track:Y даёт 75%
        #    новых. Поэтому «свежесть» (fresh=1 + queue) = станция от СЛЕДУЮЩЕГО
        #    трека → реально другой топ, а не повтор списка.
        station_id = id
        if fresh and ym_client.is_ym_id(queue or ""):
            station_id = queue
        try:
            raw, batch_id = await ym_client.rotor_tracks(
                station_id, limit=limit, queue_id=None
            )
            from .models import TrackMeta as _TM

            tracks = [_TM(**t) for t in raw]
            # фильтр уже сыгранного (queue = id сыгранного) — в fresh остаются
            # только НОВЫЕ, station_id сам не в счёт (он играющий).
            if fresh and ym_client.is_ym_id(queue or ""):
                tracks = [t for t in tracks if t.id != queue]

            # ---- ЕДИНЫЙ ЯКОРЬ (АРТИСТ > ЖАНР) + ВОЗРАСТ -----------------------
            # ЖАЛОБА 2026-10-30: ротор от seed без якоря пускал СТАРЫЙ РАНДОМ
            # (Danza Kuduro/Авария/Crazy Frog/Queen, Metallica/Evanescence от
            # Eminem). Лечим, приоритет АРТИСТ > ЖАНР (ctx «artist - title»
            # точнее общего жанра; ловил баг: ctx «MOT - Индия» → _genre_anchor
            # 'инди' обгонял артиста 'мот' → ротор без артиста не ловился):
            #   1) АРТИСТ-якорь (ctx/note дают имя): оставляем ТОЛЬКО треки этого
            #      артиста (релевантность важнее свежести — Eminem'овские Lose
            #      Yourself 2014 = не «старье», а ХИТ артиста, их НЕ режем).
            #   2) иначе ЖАНР-якорь: >=3 жанровых → ок; <3 → отбрасываем (mood ниже).
            #   3) нет якоря: режем только возраст (ротор как «Моя волна»).
            _aged = [t for t in tracks if not ym_client.is_old(t.model_dump())]
            if len(_aged) < len(tracks):
                logger.info("radio age: %d -> %d (убрано старое < %d)",
                            len(tracks), len(_aged),
                            time.gmtime(time.time()).tm_year - 5)

            def _rel(t) -> bool:
                d = t.model_dump()
                return (
                    ym_client.match_artist_anchor(d, _artist_anchor)
                    if _artist_anchor
                    else False
                )

            if _artist_anchor:
                # АРТИСТ-якорь (ПЕРВЫМ): фильтруем по артисту на СВЕЖЕМ (после
                # age), и если свежих его треков 0 — на ИСХОДНОМ (старые хиты
                # артиста допустимы, чужое — нет).
                _anchored = [t for t in _aged if _rel(t)]
                if not _anchored:
                    _anchored = [t for t in tracks if _rel(t)]   # старые хиты артиста
                if _anchored:
                    logger.info("radio artist-anchor %r: %d -> %d (артист)",
                                _artist_anchor, len(tracks), len(_anchored))
                    tracks = _anchored
                else:
                    logger.info("radio artist-anchor %r: ротор без артиста (%d/%d) -> отбрасываем",
                                _artist_anchor, len(_aged), len(tracks))
                    tracks = []   # ниже mood-поиск «<артист> подборка»
            elif _genre_anchor:
                _anchored = [t for t in _aged
                             if ym_client._genre_score(t.model_dump(), _genre_anchor) == 0]
                if len(_anchored) >= 3:
                    logger.info("radio genre-anchor %r: %d -> %d (жанр-фильтр)",
                                _genre_anchor, len(tracks), len(_anchored))
                    tracks = _anchored
                elif _aged:   # возраст уже срезал; жанр мало — оставляем свежее без жанра
                    logger.info("radio genre-anchor %r: мало жанровых (%d), держим свежее %d",
                                _genre_anchor, len(_anchored), len(_aged))
                    tracks = _aged
                else:
                    logger.info("radio genre-anchor %r: ротор без жанра (%d/%d) -> отбрасываем",
                                _genre_anchor, len(_anchored), len(tracks))
                    tracks = []   # ниже mood-поиск «<жанр> подборка»
            else:
                tracks = _aged   # нет якоря — только возраст, ротор как «Моя волна»
            logger.info(
                "radio rotor: %d tracks (station=%s fresh=%d)",
                len(tracks), station_id, fresh,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("radio rotor failed: %s", e)

        # 2) Похожие (similarTracks) — ТОЛЬКО если не fresh и Rotor дал мало.
        #    fresh=1 НЕ мешаем similar (иначе однотипные из списка лезут в поток).
        #    ЯКОРЬ (жанр/артист): если ротор отброшен (tracks=[]), similar от
        #    того же не-жанрового seed вернёт ТОТ ЖЕ мусор — пропускаем,
        #    mood-поиск ниже.
        if fresh and tracks:
            pass  # чистый Rotor-поток — разные треки, как просили
        elif _has_anchor and not tracks:
            logger.info("radio: якорь (жанр/артист), similar пропущен (мусор seed) -> mood-поиск")
        elif len(tracks) < limit:
            try:
                raw = await ym_client.similar_tracks(id, limit=limit)
                from .models import TrackMeta as _TM

                have = {t.id for t in tracks}
                tracks.extend(_TM(**t) for t in raw if t["id"] not in have)
            except Exception as e:  # noqa: BLE001
                logger.warning("radio similar failed: %s", e)

    # Нет id или Яндекс пуст → добираем подборкой по настроению (LLM), иначе — пусто.
    mood_query: str | None = None
    if len(tracks) < 5 and (note or ctx):
        # при якоре base = ЖАНР/АРТИСТ («зарубежный рэп»/«баста»), а не «похожее на <мусор-seed>»
        base = (
            (f"{note} подборка" if (note and _genre_anchor) else None)
            or (f"{_artist_anchor} подборка" if (not note or note == ctx) and _artist_anchor else None)
            or note
            or f"похожее на {ctx}"
        )
        # ФИЧА 2 (холодный старт): любимые артисты из визарда → LLM учитывает
        # ВКУС — только когда НЕТ якоря (жанра/артиста). Иначе вкусный хвост
        # ломает mood (ЛОВИЛ: «мот подборка (вкус: BastaScBastariptonith...)» →
        # LLM возвращал мусор дословно → mood-фильтр мимо → чужой рандом).
        # Также чистим мусорные токены (склейки визарда «BastaScBastari...»).
        if not _has_anchor:
            try:
                from . import taste as _tst
                _liked = [
                    a for a in _tst.get_taste()
                    if 2 <= len(a) <= 40 and not re.search(r"(.)\1{4,}", a)   # без склеек
                ]
                if _liked:
                    base = f"{base} (вкус: {', '.join(_liked[:5])})"
            except Exception:  # noqa: BLE001
                pass
        try:
            mood_query = await llm.mood_mix_query(base, current_track=ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("radio mood fallback failed: %s", e)
        # фолбэк: LLM не вернула — ищем жанр-якорь напрямую
        if not mood_query and _genre_anchor:
            mood_query = f"{_genre_anchor} подборка"
        if not mood_query and _artist_anchor:
            mood_query = f"{_artist_anchor} подборка"
        if mood_query:
            try:
                raw = await ym_client.search(mood_query, limit=limit)
                from .models import TrackMeta as _TM

                have = {t.id for t in tracks}
                # ЯКОРЬ: mood-выдача тоже фильтруем по жанру/артисту, чтобы LLM
                # не привёз мусор мимо якоря (релевантность запросу — главное).
                _mood_new = [_TM(**t) for t in raw if t["id"] not in have]
                if _has_anchor:
                    _mf = [
                        t for t in _mood_new
                        if (
                            (not _genre_anchor or ym_client._genre_score(t.model_dump(), _genre_anchor) == 0)
                            and (not _artist_anchor or ym_client.match_artist_anchor(t.model_dump(), _artist_anchor))
                        )
                    ]
                    # ЛОВИЛ: <3 → откат → Яндекс по «баста подборка» дал 2 релевантных
                    # из 10 → откат пускал 8 чужих (Баста-мусор SLŸCH/jaxxice).
                    # РЕШЕНИЕ: берём релевантные ЕСЛИ ХОТЯ БЫ 1 (пусто хуже, но чужое
                    # НИКОГДА не примешиваем при якоре) + добиваем свежими жанровыми
                    # (не артистов) чтобы не было пусто.
                    if _mf:
                        logger.info("radio mood anchor: %d -> %d (якорь)", len(_mood_new), len(_mf))
                        # добивка: свежие НЕ-артист (для «похожего» допустимо рядом),
                        # но артист-якорь ставим ПЕРВЫМИ (релевантность важнее).
                        # ФИКС (2026-10-31): _extra ОГРАНИЧЕН ≤3 (было limit-len(_mf) = 15) —
                        # иначе однотипные title-матчи (SLŸCH/Mo7cine «Basta») и случайные
                        # (ХАБИБ, Танцевальная Музыка) домешивались → жалоба «однотипные».
                        # Берём только релевантные по жанру/артисту, свежие, ≤3.
                        _mf_ids = {t.id for t in _mf}
                        _extra_pool = [t for t in _mood_new if t.id not in _mf_ids and not ym_client.is_old(t.model_dump())]
                        # при артист-якоре: title-матчи якоря допустимы («SLŸCH - Баста»),
                        # но ЧУЖИЕ без якоря (ХАБИБ/Лепс) — НЕТ. Жанр-якорь: только жанровые.
                        if _artist_anchor:
                            _extra_ok = [t for t in _extra_pool
                                         if ym_client.match_artist_anchor(t.model_dump(), _artist_anchor)
                                         or ym_client._query_genre(t.title or "") == _artist_anchor]
                        elif _genre_anchor:
                            _extra_ok = [t for t in _extra_pool
                                         if ym_client._genre_score(t.model_dump(), _genre_anchor) == 0]
                        else:
                            _extra_ok = _extra_pool
                        _extra = _extra_ok[:3]  # ≤3 добивки, не 15
                        _mood_new = _mf + _extra
                        logger.info("radio mood _extra: %d -> %d (≤3, релевантные)", len(_extra_pool), len(_extra))
                    else:
                        logger.info("radio mood anchor: 0 из %d по якорю %r (ищем шире)",
                                    len(_mood_new), _artist_anchor or _genre_anchor)
                        # пусто по якорю — берём свежие (без старья), но это уже
                        # отступление; лучше вернуть релевантные пусто.
                tracks.extend(_mood_new)
            except Exception as e:  # noqa: BLE001
                logger.warning("radio mood search failed: %s", e)

    # Фраза-обёртка: ТЗ п.9 — ПРОСТАЯ команда → ГОТОВЫЙ ШАБЛОН, без ожидания
    # LLM (раньше await до 30с откладывал СТАРТ МУЗЫКИ ради красивой фразы —
    # ТЗ п.9/п.12: «не задерживать запуск радио генерацией голосового ответа»).
    say = ("Держи ещё похожее на это." if ctx else "Подбираю похожее.")

    logger.info("radio id=%s -> %d tracks (note=%r)", id, len(tracks), note)
    # ФИКС «одни и те же» (дубли внутри батча): ротор + similar + mood могут
    # вернуть ОДНИ И ТЕ ЖЕ id — дедуплицируем, сохраняя порядок (релевантность).
    _seen_ids: set = set()
    _dedup = [t for t in tracks if not (t.id in _seen_ids or _seen_ids.add(t.id))]
    if len(_dedup) != len(tracks):
        logger.info("radio dedup: %d -> %d (убраны дубли)", len(tracks), len(_dedup))
        tracks = _dedup
    # БЛОК-ЛИСТ мусора (Macarena и т.п.) — выкидываем из /radio (mood/similar).
    _before_blk = len(tracks)
    tracks = [t for t in tracks if not ym_client._is_blocked({"title": t.title, "artist": t.artist})]
    if len(tracks) != _before_blk:
        logger.info("blocklist /radio: %d -> %d (убран мусор)", _before_blk, len(tracks))

    # ФИЛЬТР КАЧЕСТВА рекомендаций (жалоба «пошли похожие → шлак»): rotor от seed
    # отдаёт разнородный рандом — казахские мемы (Серғазы/Дана Кентай/Меломен),
    # 90-е one-hit (Mr. President «Coco Jamboo»), Chicken Banana. КАНОН Яндекса:
    # баланс новых/знакомых, разнообразие, НЕ «только лайки». Работает только когда
    # НЕТ жанр-якоря (пустой note → rotor пошёл вслепую); при note-жанре выше
    # ротор уже отброшен → здесь чистим остаток. У rotor-треков НЕТ genres/rank.
    #   1) режем 90-е one-hit по title (Coco Jamboo и т.п.);
    #   2) держим не более 2 треков одного артиста (разнообразие «Моей волны»);
    #   3) свежие наверх (year desc, неизвестный — середина).
    # Маленький пул НЕ режем до пустоты: <3 после фильтра → откат на исходный.
    if tracks:
        _pre_q = len(tracks)
        # 1) заведомые 90-е one-hit по НАЗВАНИЮ (не artist!)
        _junk_titles = {"coco jamboo"}
        _clean = [
            t for t in tracks
            if not any(w in (t.title or "").lower() for w in _junk_titles)
        ]
        # 2) лимит на артиста (сохраняем порядок = релевантность rotor)
        _seen_artist: dict = {}
        _balanced = []
        for t in _clean:
            key = (t.artist or "").strip().lower()
            if _seen_artist.get(key, 0) >= 2:
                continue
            _seen_artist[key] = _seen_artist.get(key, 0) + 1
            _balanced.append(t)
        # 3) свежесть наверх: year desc (0 = неизвестный → середина, не низ)
        if len(_balanced) >= 3:
            _balanced.sort(key=lambda t: -(t.year or 2020))
            tracks = _balanced
        else:
            tracks = _clean if len(_clean) >= 3 else tracks  # откат, не пустим
        if len(tracks) != _pre_q:
            logger.info(
                "radio quality: %d -> %d (однородность, ≤2/артист, свежесть)",
                _pre_q, len(tracks),
            )

    # ФИЧИ 1+4: обратная связь юзера — РЕКОМЕНДАЦИИ (radio/similar/rotor/refill)
    # фильтруются и по дизлайкам, и по скипам (1 скип = «не это», 3 скипа артиста
    # = бан артиста). В точном поиске (api_search) — только дизлайки.
    from . import taste as _taste
    tracks = _taste.filter_tracks(tracks)

    return SearchResponse(
        query=f"radio:{id or note or ctx}",
        results=tracks[: config.MAX_SEARCH_RESULTS],
        total=len(tracks),
        intent="playlist",
        mix=True,
        say=say,
        mood_query=mood_query,
        batch_id=batch_id,
    )


# ---------------------------------------------------------------------------
# POST /radio/feedback {id, batch_id, type} — петля Rotor: сообщаем «трек
# стартовал», чтобы следующий /radio?fresh=1 вернул СВЕЖИЙ батч (другие треки,
# не тот же список). Молча 200 при недоступности (рекомендации и так идут).
# ---------------------------------------------------------------------------
class RadioFeedbackIn(_BM):
    id: str = Field(..., max_length=64)          # ym_<id> трека
    batch_id: str = Field("", max_length=128)
    type: str = Field("trackStarted", max_length=32)   # trackStarted|like|dislike|skip|completed
    artist: str = Field("", max_length=200)            # для skip-бана артиста
    # ТЗ п.6.4: доля прослушивания 0..1 для взвешенных реакций;
    # -1 = клиент не прислал (обратная совместимость, прежнее поведение).
    progress: float = Field(-1.0, ge=-1.0, le=1.0)
    # 2026-11-01: как переключили — manual (ручная «следующая») / auto
    # (быстрый автопереход) / '' (неизвестно). Разная сила skip: см. taste.record.
    context: str = Field("", max_length=16)


@app.post("/radio/feedback")
async def api_radio_feedback(body: RadioFeedbackIn):
    """ФИЧИ 1+4: обратная связь плеера.
    trackStarted/like → rotor Яндекса (проверено: dislike/skip там 400 →
    локально через taste) + like/dislike/skip персистятся в taste-профиле
    и фильтруют будущие рекомендации (см. taste.filter_tracks)."""
    from . import ym_client, taste

    ok = False
    kind = (body.type or "trackStarted").strip()
    # локальный профиль вкуса (всегда, даже без batch_id)
    if kind in ("like", "unlike", "dislike", "skip", "completed"):
        try:
            from . import metrics as _mt_fb
            _mt_fb.event(f"fb_{kind}")
        except Exception:  # noqa: BLE001
            pass
        # ТЗ п.3: жанры трека подтягиваем из истории (ym-треки записали
        # genres при выдаче) — для профиля предпочитаемых жанров.
        _g: list[str] = []
        try:
            from . import history as _h_g
            _g = list((_h_g.track_meta(body.id) or {}).get("genres") or [])
        except Exception:  # noqa: BLE001
            pass
        taste.record(kind, body.id, body.artist, progress=body.progress,
                     genres=_g, context=(body.context or "").strip()[:16])
    # ТЗ п.3: история прослушиваний — plays/last_play для «последних 20»
    # и метаданные трека (id → артист/название) для контекста.
    try:
        from . import history as _hist_f
        if kind == "completed":
            _hist_f.record_play(body.id)
        if body.id and body.artist:
            _hist_f.record_tracks_meta([{
                "id": body.id, "artist": body.artist,
                "title": getattr(body, "title", "") or "",
            }])
    except Exception:  # noqa: BLE001
        pass
    # petля ротора: trackStarted (свежий батч) + like (вкус станции)
    if body.batch_id and ym_client.is_ym_id(body.id) and kind in ("trackStarted", "like"):
        try:
            ok = await ym_client.rotor_feedback(body.id, body.batch_id, kind)
        except Exception as e:  # noqa: BLE001
            logger.info("radio feedback error: %s", e)
    return {"ok": ok or kind in ("like", "unlike", "dislike", "skip")}


# ------------------------------------------------------------------ taste ---
# ФИЧА 2: визард холодного старта (APK при первом запуске) → любимые артисты,
# сервер подмешивает в mood-подборки (см. api_radio mood-fallback).
class TasteIn(_BM):
    artists: list[str] = Field(default_factory=list, max_length=30)


@app.post("/api/taste")
async def api_set_taste(body: TasteIn):
    from . import taste
    saved = taste.set_taste([a[:80] for a in body.artists if (a or "").strip()])
    return {"ok": True, "artists": saved}


# ТЗ п.14: метрики качества (понимание команд vs рекомендации) — см. app/metrics.py
@app.get("/api/metrics")
async def api_metrics():
    from . import metrics as _mt
    return _mt.snapshot()


@app.get("/api/taste")
async def api_get_taste():
    from . import taste
    return taste.stats()


# GET /api/explain?id=...&q=... — ОБЪЯСНЕНИЕ рекомендации (ТЗ п.14).
# Причины строго из РЕАЛЬНЫХ данных трека (п.19: не выдумывать признаки):
# совпадение жанров с запросом, якорь исполнителя, свежесть, наличие в
# вкусе пользователя, наличие популярности в источнике.
@app.get("/api/explain")
async def api_explain(
    id: str = Query("", max_length=300),   # ma_-id длиннее 64 (URL-энкод из MA)
    q: str = Query("", max_length=200),
    ctx: str = Query("", max_length=300),
):
    from . import taste as _tex
    from . import ym_client as _yx

    if not id:
        raise HTTPException(status_code=400, detail="id required")
    # Ищем трек в источниках (по id не везде есть точный lookup → берём из
    # свежего поиска по запросу; если нет — объясняем по метаданным id=).
    track: dict = {}
    try:
        raw = await _yx.search(q or ctx or id, limit=10) if (q or ctx) else []
        for t in raw:
            if t.get("id") == id:
                track = t
                break
    except Exception:  # noqa: BLE001
        track = {}
    if not track:
        track = {"id": id, "artist": "", "title": "", "year": 0, "genres": []}

    reasons: list[str] = []
    artist = (track.get("artist") or "").lower()
    genres = [str(g).lower() for g in (track.get("genres") or [])]
    # 1) якорь исполнителя из запроса
    q_norm = _yx._norm_artist(q) if q else ""
    if q_norm and any(tok in artist for tok in q_norm.split() if len(tok) >= 4):
        reasons.append("исполнитель совпадает с запросом")
    # 2) жанры пересекаются с запросом
    q_tokens = [w for w in (q or "").lower().replace("ё", "е").split() if len(w) >= 3]
    if genres and q_tokens:
        _g_blob = " ".join(genres)
        if any(w in _g_blob for w in q_tokens):
            reasons.append(f"жанр трека ({', '.join(genres[:3])}) совпадает с запросом")
    # 3) свежесть (реальный год)
    y = int(track.get("year") or 0)
    import time as _t
    cy = _t.gmtime(_t.time()).tm_year
    if y:
        reasons.append(f"год выпуска {y}" + (" — новинка" if y >= cy - 2 else ""))
    # 4) в личном вкусе
    if artist and any(a and a in artist for a in _tex.get_taste()):
        reasons.append("исполнитель в вашем вкусе")
    # 5) популярность в источнике (реальный likesCount, если заполнен)
    pop = int(track.get("popularity") or 0)
    if pop > 0:
        reasons.append(f"популярен в источнике (лайков альбома: {pop})")
    if not reasons:
        reasons.append("подобрано по похожести метаданных с запросом")

    return {"id": id, "track": track, "reasons": reasons}


# POST /smart  {text, ctx}  — ГЛАВНЫЙ диалог с ассистентом («как Алиса»).
# LLM САМА управляет: решает инструмент (play/radio/mood/next/stop/resume)
# и говорит фразой. Сервер нормализует tool_calls → структурированный `action`,
# который приложение исполняет 1:1 (APK не гадает по именам функций).
# Возвращает {say, action, tool_calls}.
# ---------------------------------------------------------------------------
class SmartIn(_BM):
    text: str = Field(..., min_length=1, max_length=500)
    ctx: str = Field("", max_length=300)  # текущий трек («артист - название»)


# ------------------------------------------------------------------
# Живая погода (без ключей): wttr.in → Open-Meteo (фолбэк).
# LLM без данных всегда отвечает «нет доступа к погоде» — подставляем реальную.
# ------------------------------------------------------------------
_WMO = {0: "ясно", 1: "малооблачно", 2: "переменная облачность", 3: "пасмурно",
        45: "туман", 48: "изморозь", 51: "морось", 53: "морось", 55: "морось",
        61: "дождь", 63: "дождь", 65: "сильный дождь", 71: "снег", 73: "снег",
        75: "сильный снег", 80: "ливень", 81: "ливень", 82: "сильный ливень",
        95: "гроза", 96: "гроза", 99: "сильная гроза"}


def _extract_city(text: str) -> str:
    """Достать город из запроса («погода в Новосибирске» → «Novosibirsk» паттерн)."""
    t = text.lower()
    m = re.search(r"\bв\s+([а-яё\-]+)\b", t)
    return m.group(1) if m else ""


async def _weather_for(text: str) -> str | None:
    """Реальная погода, если запрос про погоду. None — не про погоду / недоступно."""
    t = text.lower()
    if "погод" not in t and "температур" not in t:
        return None
    city = _extract_city(t) or "Novosibirsk"   # по умолчанию — Новосибирск
    try:
        async with httpx.AsyncClient(timeout=6) as c:
            # 1) wttr.in (простой, без ключа)
            r = await c.get(f"https://wttr.in/{city}", params={"format": "3", "lang": "ru"})
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
            # 2) Open-Meteo фолбэк: координаты не угадываем — дефолт Новосибирск
            r = await c.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": 55.03, "longitude": 82.92,
                        "current": "temperature_2m,weather_code",
                        "timezone": "Asia/Novosibirsk"},
            )
            if r.status_code == 200:
                cur = r.json().get("current", {})
                temp = cur.get("temperature_2m")
                code = cur.get("weather_code")
                if temp is not None:
                    desc = _WMO.get(code, "")
                    return f"{temp}°C {desc}".strip()
    except Exception as e:  # noqa: BLE001 — погода не должна ронять /smart
        logger.warning("weather failed: %s", e)
    return None


def _llm_say_enabled() -> bool:
    """Разрешает ли LLM переписывать шаблонную фразу say.

    P0: шаблон ВСЕГДА готов заранее (мгновенно), LLM только «оживляет».
    Если LLM выключен/недоступен — остаётся шаблон (никакой задержки).
    """
    from .config import config as _cfg
    return bool(_cfg.LLM_ENABLED and _cfg.LLM_API_KEY)


def _clean_say(text: str | None) -> str | None:
    """Убрать из фразы озвучки сырой tool-call / код-блоки LLM.

    Жалоба: в озвучке засветилось «```play_music(query="ANNA ASTI ...")```».
    Модель отдаёт и фразу, и вызов инструмента в content — для say годится
    ТОЛЬКО человеческая часть. Вырезаем тройные кавычки-блоки, inline-код
    и голые вызовы функций (play_music(...) и т.п.).
    """
    if not text:
        return text
    s = text
    # --- ЖЁСТКИЙ срез: от первого СЛУЖЕБНОГО маркера до конца -------------
    # Модель (особенно без tools в /search) приклеивает хвост: тег
    # ...play_music>, вызов <parameter>..., нумерованный список треков.
    # Режем по ПЕРВОМУ такому маркеру — человеческая фраза всегда в начале.
    markers = [
        "\n\nНайдено музыка", "\nНайдено музыка", "Найдено музыка",
        "\n1. ", "\n2. ", "\n\n1.", "<parameter", "<function",
        "<parameter>", "play_music>", "play_radio>", "next_music>",
        "\n\nВыберите номер",
    ]
    cut = -1
    for mk in markers:
        i = s.find(mk)
        if i != -1:
            cut = i if cut == -1 else min(cut, i)
    if cut != -1 and cut > 0:
        s = s[:cut]
    # --- остатки markdown-блоков / кода ----------------------------------
    b = s.find("```")
    if b != -1:
        has_nl = "\n" in s[b:]
        if has_nl:
            s = s[:b]          # БЛОК (есть перевод строки) — резать до конца
        else:
            s = s.replace("```", "`")   # inline-triple «```x```» → «`x`» (снять как inline)
    s = re.sub(r"`([^`\n]+)`", r" \1 ", s)   # inline: раскрыть кавычки, текст оставить
    s = re.sub(r"`+", " ", s)
    # вырезать inline-имена инструментов, оставшиеся после раскрытия кавычек
    s = re.sub(r"\b(play_music|play_radio|next_music|stop_music|resume_music)\b", " ", s)
    s = re.sub(r"</?[A-Za-z][A-Za-z0-9_]{1,30}(\s+[^>\n]*)?>", " ", s)
    s = re.sub(r"`([A-Za-z0-9_]{1,20})`", r" \1 ", s)
    s = re.sub(r"`([^`\n]*[А-Яа-яё][^`\n]*)`", r" \1 ", s)
    s = re.sub(r"`+", " ", s)
    s = re.sub(r"(?<![A-Za-z0-9])[A-Za-z_][A-Za-z0-9_]*\s*=\s*\S+", " ", s)
    # хвосты из угловых скобок: убрать одиночные <> и закольцованные <...>-остатки.
    # НЕ трогаем честные "<3", "2<3", "[0]", "a<b" (прилегание без пробелов).
    s = re.sub(r"<\s+", " ", s)              # «...отдых. < музыка» → убрать «< »
    s = re.sub(r"\s+>", " ", s)               # «музыка <» → убрать « <»
    s = re.sub(r"<[^<>]{0,40}>", " ", s)      # закольцованный остаток «<...>»
    s = re.sub(r"[\s,.:;!?\-–—]*[<>][\s,.:;!?\-–—]*$", " ", s)
    # чужие алфавиты (кит. иероглифы и т.п.), которые модель переслаёт из сниппетов:
    # «...двух дней» / «休息» → оставить только латиницу/кириллицу/цифры/знаки.
    s = re.sub(r"[^\x00-\x7FА-Яа-яёЁ\d\s,.:;!?%°\-–—()/+\"'«»№#]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—:\t\"'()[]{}«»<>")
    # хвостовой мусор-стоп: СНАЧАЛА strip (убрал trailing-пробел из-за которого
    # раньше не матчился хвост), ПОТОМ срез «... отдых. музыка»
    if len(s.split()) >= 5 and re.search(r"[\s.]музыка$", s):
        s = re.sub(r"[\s.]музыка$", "", s).strip(" -–—:\t\"'")
    return s or None


def _action_from_tool(tool_calls: list) -> dict | None:
    """tool_calls LLM → действие для приложения (детерминированный маппинг)."""
    if not tool_calls:
        return None
    tc = tool_calls[0]
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:  # noqa: BLE001
        args = {}
    if name == "play_music":
        query = (args.get("query") or "").strip()
        mood = (args.get("mood") or "").strip()
        if mood:
            # подборка по настроению: настроение — основа запроса
            q = f"{mood} {query}" if query and query.lower() not in ("музыка", "music") else mood
            return {"type": "search", "query": q, "mix": 1}
        if query:
            return {"type": "search", "query": query, "mix": 0}
        return None
    if name == "play_radio":
        return {"type": "radio", "note": (args.get("note") or "").strip()}
    if name == "next_music":
        return {"type": "next"}
    if name == "stop_music":
        return {"type": "stop"}
    if name == "resume_music":
        return {"type": "resume"}
    return None


# ТЗ п.10.8/п.13.12: защита от ДВОЙНОЙ отправки команды (дабл-тап голосом /
# кнопкой): тот же текст в коротком окне → бездействуем. Иначе «следующий»
# ×2 = два перелистывания. Глобальное окно процесса (один пользователь).
_LAST_CMD: tuple = ("", 0.0)
# Текущая команда пользователя (для _action_from_tool: отличить настроение,
# НАЗВАННОЕ в просьбе, от слова из LLM-контекста/истории — ТЗ п.13 фикс мусора)
_LAST_CMD_TEXT: str = ""
_DUP_WINDOW_SEC = 3.0


@app.post("/smart")
async def api_smart(body: SmartIn):
    from . import metrics as _mt
    _mt.event("smart_total")
    # ТЗ п.10.8: дедуп САМЫМ ПЕРЫМ (до fastpath и LLM) — тот же текст ≤3с
    # → без действия (дабл-тап «следующий» ×2 = два перелистывания).
    global _LAST_CMD
    _now_cmd = time.time()
    if (_LAST_CMD[0] == body.text
            and (_now_cmd - _LAST_CMD[1]) <= _DUP_WINDOW_SEC):
        logger.info("smart dedup: повтор %r за %.1fс -> action=null",
                    body.text[:40], _now_cmd - _LAST_CMD[1])
        _mt.event("smart_dedup")
        return {"say": "Уже делаю.", "action": None, "tool_calls": None}
    _LAST_CMD = (body.text, _now_cmd)
    global _LAST_CMD_TEXT
    _LAST_CMD_TEXT = body.text
    context = f"Играет: {body.ctx}" if body.ctx else ""
    # ТЗ п.4 «краткосрочный контекст»: последние 10 прослушанных + последние
    # 3 запроса — ограниченный размер (не вся история). LLM понимает
    # «продолжи в том же духе» / «вчера мне понравилось».
    try:
        from . import history as _h_ctx
        _r10 = _h_ctx.recent(10)
        if _r10:
            _lst = ", ".join(
                f"{(t.get('artist') or '')[:25]} - {(t.get('title') or '')[:30]}"
                for t in _r10
            )
            context = (context + " | " if context else "") + f"Недавно играло: {_lst}"
        _s3 = _h_ctx.last_searches(3)
        if _s3:
            context += " | Недавние запросы: " + ", ".join(s["q"][:30] for s in _s3)
        context = context[:600]  # лимит размера контекста (ТЗ п.4/п.10)
    except Exception:  # noqa: BLE001
        pass
    logger.info("smart: %r (ctx=%r)", body.text, body.ctx)   # диагностика голоса

    # ── P0 FAST-PATH: простые команды БЕЗ LLM (мгновенно, бесплатно) ─────
    # «следующий», «пауза», «громче», «включи музыку» — детерминированно.
    from . import fastpath as _fp
    _fast = _fp.try_fast_path(body.text) or _fp.try_simple_music(body.text)
    if _fast:
        logger.info("smart fast-path: %r -> %s (без LLM)", body.text[:40], _fast["action"].get("type"))
        _mt.event("smart_fastpath")
        return {"say": _fast["say"], "action": _fast["action"], "tool_calls": None}

    # ТЗ п.7 «персональный помощник»: реакции на КАЧЕСТВО подборки —
    # детерминированно и без LLM (п.10: быстро, бесплатно, надёжно).
    # note сквозной: APK передаёт его в /radio → сервер включит разнообразие
    # (mpa=1 + анти-повтор), не ломая жанр-якорь (note = смысловая подсказка).
    _t_q = body.text.strip().lower()
    if re.search(
            r"однообраз|одинаков|заелось|одно и то же|одни и те же|разнообраз|"
            r"не повтор|не повторяй|повторяе|новых артист|другие артисты", _t_q):
        logger.info("smart quality-reaction: %r -> radio (diverse)", body.text[:50])
        _mt.event("smart_quality_reaction")
        return {
            "say": "Сделаю разнообразнее — добавлю других артистов.",
            "action": {"type": "radio", "note": "разнообразнее, новые артисты, без повторов"},
            "tool_calls": None,
        }
    if re.search(r"уже\s+слышал|это\s+зна|повторяешь|заранее\s+знаешь", _t_q):
        logger.info("smart quality-reaction: %r -> radio (fresh)", body.text[:50])
        _mt.event("smart_quality_reaction")
        return {
            "say": "Принял — предложу новое, этого не будет.",
            "action": {"type": "radio", "note": "новое, свежее, этого артиста не предлагать"},
            "tool_calls": None,
        }

    # ТЗ п.7/п.13 «продолжи в том же духе/настроении» → РАДИО от текущего
    # трека ДЕТЕРМИНИРОВАННО: LLM нестабильно отдавала то radio, то
    # search-mix (дребезжание видно в прогонах тестов п.13).
    if body.ctx and re.search(
            r"(продолж\w*\s+(в\s+том\s+же|в\s+таком\s+же)"
            r"|в\s+том\s+же\s+(дух|настроен)|в\s+таком\s+же\s+(ключ|дух))", _t_q):
        logger.info("smart continue-style: %r -> radio", body.text[:50])
        _mt.event("smart_quality_reaction")
        return {
            "say": "Продолжаю в том же духе.",
            "action": {"type": "radio", "note": "продолжай в том же настроении"},
            "tool_calls": None,
        }

    # ТЗ п.8 «текстовая обратная связь → валидируемая операция профиля»:
    # «этот исполнитель надоел / не нравится X» → вес артиста вниз с
    # источником и журналом событий (серверная операция, LLM к БД не
    # прикасается — перехват детерминированный, action не требуется).
    if re.search(r"надоел|надоела|достал|достала|больше\s+не\s+хочу|"
                 r"запел|запела|не\s+нравится", _t_q):
        _a_pref = ""
        _m_a = re.search(
            r"(?:не\s+нравится|надоел[аи]?|достал[аи]?)\s+"
            r"([а-яёa-z][а-яёa-z\s\-'’]{2,30})", _t_q)
        if _m_a:
            _a_pref = _m_a.group(1).strip(" -–,.;:!?")
        if not _a_pref and body.ctx and " - " in body.ctx:
            _a_pref = body.ctx.split(" - ")[0].strip()
        if len(_a_pref) >= 3:
            try:
                from . import taste as _t_pref
                _w_after = _t_pref.record_artist_preference(
                    _a_pref, -2.0, source="assistant-text")
                logger.info("smart text-feedback: %r -> artist %r w=%.2f",
                            body.text[:50], _a_pref, _w_after)
                _mt.event("smart_text_feedback")
                return {
                    "say": f"Учту: {_a_pref} — буду реже его предлагать.",
                    "action": None,   # профиль обновлён, плеер не трогаем
                    "tool_calls": None,
                }
            except Exception as _e_pf:  # noqa: BLE001
                logger.warning("text-feedback failed: %s", _e_pf)

    # Живая погода (LLM без данных всегда «нет доступа») — даём ей контекст,
    # чтобы обернула в живую фразу. Иначе фолбэк-ответ ниже.
    # (дедуп команд — САМЫМ первым в api_smart, см. ТЗ п.10.8)
    weather = await _weather_for(body.text)
    if weather:
        context = (context + " | " if context else "") + f"Текущая погода: {weather}"
    r = await llm.smart_response(body.text, context=context, tools=True,
                                 timeout=12.0)   # ТЗ п.10: таймаут вместо 30с
    say = _clean_say(r.get("text"))   # без сырого «```play_music(...)```»
    tool_calls = r.get("tool_calls")
    action = _action_from_tool(tool_calls)
    # ТЗ п.5 «не допускать action:null для корректно распознанной команды»:
    # LLM ответила текстом, но не вызвала инструмент → безопасный детерминированный
    # фолбэк: явная команда на воспроизведение → search с этим текстом.
    if action is None:
        _t_norm = body.text.strip().lower()
        _m_cmd = re.match(
            r"^(?:а\s+|ну\s+|давай\s+|можно\s+)?"
            r"(включи|включить|поставь|поставить|запусти|запустить"
            r"|подбери|подобрать|найди|найти|поищи|поискать)\s+(.{3,})$",
            _t_norm,
        )
        if _m_cmd:
            _q_fb = _m_cmd.group(2).strip()
            # чистим хвосты-подсказки
            _q_fb = re.sub(r"^(что-нибудь\s+|мне\s+|пожалуйста\s+)", "", _q_fb)
            if len(_q_fb) >= 3:
                _mix_fb = 1 if re.search(
                    r"подборк|микс|что-нибудь|что нибудь|похож|лучш|хит|чарт", _q_fb) else 0
                action = {"type": "search", "query": _q_fb[:120], "mix": _mix_fb}
                logger.info("smart action-null fallback: %r -> search mix=%d",
                            _q_fb[:50], _mix_fb)
                _mt.event("smart_null_fallback")
                if not say:
                    say = "Включаю." if not _mix_fb else "Собираю подборку."
    # music_search: LLM нашла треки напрямую, но НЕ вызвала play_music (лимит
    # раундов/запуталась) → force search с ТОЧНЫМ music_search-запросом (напр.
    # «Stranger Things OST»), а НЕ с «5 треков из сериала...» (его /search
    # понимает плохо → мусор). /search с точным запросом даст настоящие треки.
    if action is None and r.get("music_searched"):
        _q_ms = (r.get("last_ms_query") or "").strip() or body.text.strip()
        _q_ms = _q_ms[:120]
        action = {"type": "search", "query": _q_ms, "mix": 0}
        logger.info("smart music_search force-search: %r (last_ms=%r)",
                    _q_ms, r.get("last_ms_query"))
        _mt.event("smart_ms_force_search")
        if not say:
            say = "Нашла треки — включаю."
    # Если LLM вызвал инструмент, но фразы нет — короткая заглушка по типу.
    if not say and action:
        say = {
            "search": "Включаю.",
            "radio": "Подбираю похожее.",
            "next": "Слушаем следующий.",
            "stop": "Останавливаю.",
            "resume": "Продолжаем.",
        }.get(action["type"], "Хорошо.")
    if not say:
        say = "Не совсем понял, повтори иначе."
    # Фолбэк: погода, но LLM ответила отказом («нет доступа») — отдать реальную.
    if weather and say and re.search(r"нет\s+(доступа|данных|погод)|не\s+(могу|вижу|подскажу)", say, re.I):
        say = f"Сейчас {weather}."
    # Фолбэк ВРЕМЕНИ (по Новосибирску): запрос про часы/время, а ответ пуст/заглушка
    # (LLM изредка не формулирует) → отдать точное локальное время детерминированно.
    tl = body.text.lower()
    wants_time = (("час" in tl or "время" in tl) and len(tl.split()) <= 4)
    if wants_time and (not say or "Не совсем понял" in say):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Novosibirsk"))
        say = f"Сейчас в Новосибирске {now:%H:%M}, {now:%d.%m.%Y}."
        logger.info("time-fallback injected in /smart")
    if action is None:
        _mt.event("smart_action_null")
    return {"say": say, "action": action, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Crash-log intake from the Android app (diagnosis without adb).
#   POST /api/applog   {device, tail}  — phone uploads its log after a crash
#   GET  /api/applog   — agent reads the latest uploaded log
# ---------------------------------------------------------------------------
from pydantic import BaseModel as _BM

LOG_DIR = Path(config.DATA_DIR) / "applogs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LATEST_LOG = LOG_DIR / "latest.log"


class AppLogIn(_BM):
    device: str = "unknown"
    tail: str = ""


@app.post("/api/applog")
async def api_receive_applog(body: AppLogIn):
    text = (body.tail or "")[-200_000:]
    if not text:
        raise HTTPException(status_code=400, detail="empty log")
    stamped = LOG_DIR / f"{int(time.time())}.log"
    header = f"# device={body.device} received={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    stamped.write_text(header + text, encoding="utf-8")
    LATEST_LOG.write_text(header + text, encoding="utf-8")
    logger.info("applog received from %s (%d bytes)", body.device, len(text))
    return {"status": "stored", "bytes": len(text)}


@app.get("/api/applog")
async def api_read_applog():
    if not LATEST_LOG.exists():
        raise HTTPException(status_code=404, detail="no logs yet")
    return {"log": LATEST_LOG.read_text(encoding="utf-8")}
