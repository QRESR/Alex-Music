"""Configuration for Alex Music."""

import os
from dataclasses import dataclass, field
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # yt-dlp
    YT_DLP_PATH: str = "yt-dlp"
    YT_DLP_ARGS: list = field(default_factory=lambda: [
        "--no-warnings",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "best",
        "--audio-quality", "0",  # best quality
    ])

    # YouTube cookies (server IP is bot-blocked without them)
    COOKIES_FILE: str = str(_BASE_DIR / "private" / "youtube-cookies.txt")
    # Куки авторизованной сессии VK Музыки (третий источник треков).
    VK_COOKIES_FILE: str = str(_BASE_DIR / "private" / "vk-cookies.txt")

    # Persistent data (playlists)
    DATA_DIR: str = str(_BASE_DIR / "data")

    # Static frontend
    STATIC_DIR: str = str(_BASE_DIR / "static")

    # JS runtime for YouTube's signature/n-challenge solver.
    # yt-dlp enables only "deno" by default; server has node instead.
    # Format: {name: {"path": ...}} — matches yt-dlp's js_runtimes param.
    JS_RUNTIMES: dict = field(default_factory=lambda: {"node": {"path": None}})

    # Cache TTL (seconds)
    SEARCH_CACHE_TTL: int = 1800      # 30 min
    STREAM_URL_CACHE_TTL: int = 14400  # 4 hours (YouTube URLs expire ~6h)
    METADATA_CACHE_TTL: int = 3600     # 1 hour

    # Streaming
    STREAM_CHUNK_SIZE: int = 65536  # 64KB chunks
    PROXY_TIMEOUT: float = 30.0     # timeout for proxied requests

    # Limits
    MAX_SEARCH_RESULTS: int = 20

    # ── LLM (Xiaomi MiMo) — «мозг Алисы» ────────────────────────────────
    # Чтение из env (может быть в EnvironmentFile systemd). Дефолты
    # воспроизводят РАБОЧУЮ комбинацию (проверено: HTTP 200):
    #   ключ=tp-e... из profiles/gera/.env, хост=token-plan-ams, model=v2.6-flash.
    LLM_BASE_URL: str = os.environ.get(
        "LLM_BASE_URL", "https://token-plan-ams.xiaomimimo.com/v1"
    )
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")  # ПУСТО, если нет env
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "mimo-v2.6-flash")
    LLM_TIMEOUT: float = 30.0
    LLM_ENABLED: bool = os.environ.get("LLM_ENABLED", "1") == "1"
    # ТЗ п.11 «архитектура нескольких моделей»: FAST-провайдер для простых
    # задач (say-фразы, настроение). ПУСТОЙ по умолчанию → работает только
    # основной Mimo (НЧЕГО не меняется, пока не настроен вторым провайдером).
    # Advanced = основной Mimo; Local = детерминированные фолбэки (fastpath,
    # offline-парсер) — уже работают без сети.
    LLM_FAST_BASE_URL: str = os.environ.get("LLM_FAST_BASE_URL", "")
    LLM_FAST_API_KEY: str = os.environ.get("LLM_FAST_API_KEY", "")
    LLM_FAST_MODEL: str = os.environ.get("LLM_FAST_MODEL", "")

    # ── RECOMMENDATION ENGINE (патч, feature-флаг) ─────────────────────────
    # False = работает СТАРЫЙ _rank_by_artist() (дефолт, ничего не меняется).
    # True  = новый recommender.rank() с весами/свежестью/разнообразием.
    # Откат: выставить False и перезапустить music-proxy (код не удалять).
    USE_NEW_RANKER: bool = os.environ.get("USE_NEW_RANKER", "0") == "1"
    # ТЗ «Похожие» 2026-10-31: переработанный mode=similar (параллельный сбор +
    # CLAP + единый rank + MMR). 0 → прежний код в api_radio (откат).
    USE_SIMILAR_V2: bool = os.environ.get("USE_SIMILAR_V2", "0") == "1"
    # Тип запроса для весов: auto определяется по mix/ctx. Может быть вручную.
    # max_per_artist: предел треков одного артиста в выдаче (0 = выкл).
    RECOMMENDER_MAX_PER_ARTIST: int = int(os.environ.get("REC_MAX_PER_ARTIST", "3"))
    # Лимит источников-кандидатов перед ранжированием (ширина сбора).
    REC_CANDIDATE_MULT: int = int(os.environ.get("REC_CANDIDATE_MULT", "2"))

    # ── Внешние базы метаданных (P2, кэш обязателен) ──────────────────────
    # MusicBrainz: алиасы артистов, теги/жанры, определение cover (без ключа).
    # Лимит 1 req/sec — держим свой ритм. Ключ не нужен.
    MB_ENABLED: bool = os.environ.get("MB_ENABLED", "1") == "1"
    MB_CACHE_TTL: int = int(os.environ.get("MB_CACHE_TTL", str(7 * 24 * 3600)))  # 7 дней
    MB_MIN_INTERVAL: float = 1.1  # сек между запросами (лимит MusicBrainz)

    # ── TTS (edge-tts, русские голоса) — озвучка ответов как у Алисы ─────
    TTS_VOICE: str = os.environ.get("TTS_VOICE", "ru-RU-SvetlanaNeural")
    TTS_ENABLED: bool = os.environ.get("TTS_ENABLED", "1") == "1"
    TTS_SPEED: float = float(os.environ.get("TTS_SPEED", "1.0"))
    TTS_CACHE_DIR: str = str(_BASE_DIR / "data" / "tts")  # кэш mp3


config = Config()
