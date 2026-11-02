"""Pydantic models for API responses."""

from pydantic import BaseModel, Field


class TrackMeta(BaseModel):
    """Metadata for a single track."""
    id: str = Field(..., description="YouTube video ID")
    title: str
    artist: str = ""
    duration: int = Field(0, description="Duration in seconds")
    thumbnail: str = ""
    channel: str = ""
    url: str = ""  # original YouTube URL
    # Год релиза (0 = неизвестен) — для клиентского ранжирования по свежести
    # (жалоба «собирает старые шлягеры» — клиент поднимает новее наверх).
    year: int = 0
    # Позиция в выдаче Яндекса (proxy популярности; 0 если источник не задал
    # порядок) — для гибридного скоринга «свежесть × популярность».
    rank: int = 0
    # ---- Патч 2, шаг 4 (ТЗ п.3.3 Feature Extraction) -----------------------
    # Заполняются источником, ЕСЛИ доступны; пустые значения = «неизвестно»
    # (ТЗ п.4: не выдумывать недостающие характеристики).
    genres: list[str] = []             # жанры из артиста/альбома (Яндекс)
    album: str = ""                    # альбом
    explicit: bool = False             # 18+
    popularity: int = 0                # likesCount альбома (Яндекс) или 0
    listening_ratio: float = 0.0       # listeningFinished (коллективный сигнал)
    loudness: float = 0.0              # r128 LRA/level (Яндекс), 0 = неизвестен


class SearchResponse(BaseModel):
    """Search results (with optional Alice-like intent metadata)."""
    query: str
    results: list[TrackMeta]
    total: int
    # Intent metadata (present on /search): play | playlist | suggest
    intent: str = "play"
    mix: bool = False
    playlist: str | None = None  # matched stored playlist name, if any
    # Alice-like extras: короткая фраза для озвучки («Включаю подборку по настроению»)
    # и фактический поисковый запрос подборки, если LLM его уточнил по настроению.
    say: str | None = None
    mood_query: str | None = None
    # Rotor-петля: batch_id текущего батча рекомендаций → клиент шлёт feedback
    # (trackStarted), чтобы следующий /radio вернул СВЕЖИЙ батч (другие треки).
    batch_id: str | None = None


class StreamInfo(BaseModel):
    """Internal stream info (not exposed to client)."""
    video_id: str
    direct_url: str  # YouTube CDN URL
    http_headers: dict = Field(default_factory=dict)
    expires_at: float = 0  # unix timestamp


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
