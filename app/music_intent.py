"""MusicIntent: структурированный разбор музыкального запроса (P1 ТЗ п.4.4).

Зачем: вместо преобразования просьбы в ОДНУ строку LLM возвращает структурированный
intent (mood/activity/energy/genres/freshness/...), из которого сервер сам строит
несколько поисковых запросов и собирает кандидатов.

Контракт:
    parse_intent_llm(text, current_track="") -> MusicIntent | None
    MusicIntent.to_search_queries() -> list[str]   (несколько вариантов)
    MusicIntent.to_rec_kwargs() -> dict            (параметры для recommender)

Совместимость: если LLM недоступна или вернул мусор — возвращаем None,
вызывающий код использует СТАРЫЙ mood_mix_query (текстовую строку).
НИКОГДА не роняет обработку.

Проверено живьём (эксперимент 2026-10-30):
  - Xiaomi поддерживает response_format json_object и json_schema.
  - tools + response_format ВМЕСТЕ ломают tool_calls → вызываем ОТДЕЛЬНО.
  - temperature=0.1 для детерминизма.
"""
from __future__ import annotations

import copy as _copy
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("music_intent")

# Контекст предыдущих команд (ТЗ п.13: «Контекст предыдущих команд»).
# Хранится в памяти процесса; хранит последний разобранный intent.
_LAST_INTENT: dict | None = None
# ТЗ п.10.11: кэш LLM-разборов (≤32, LRU-вытеснение по вставке).
_INTENT_CACHE: dict = {}


def _prev_context() -> str:
    """Строка контекста для LLM: что делали до этого (последний intent)."""
    global _LAST_INTENT
    if not _LAST_INTENT:
        return ""
    li = _LAST_INTENT
    return (
        f"Предыдущая команда: «{li.get('text', '')}» → action={li.get('action')}, "
        f"mood={li.get('mood')}, genres={li.get('genres')}, "
        f"seed={li.get('seed_artists')}."
    )


# ТЗ п.4 «чёткие правила обновления контекста»: прошлая команда подмешивается
# ТОЛЬКО в УТОЧНЕНИЯ к ней, а не в новые независимые запросы. Ловил баг:
# после «Асти модный поп» запрос «танцевальная подборка» унаследовал
# seed='Asti' и decade=1990-х (дрейф контекста, журнал 09:52-09:53).
_FOLLOWUP_RE = re.compile(
    r"(ещ[её]\b|тоже\b|то\s+же|такой\s+же|такая\s+же|поспокой|поживее|покруче"
    r"|помедленнее|потише|вместо|только\s+не|другой|другого|похож|продолж"
    r"|опять|снова|в\s+том\s+же|верни\s+предыдущ|назад\s+к|продолжай)",
    re.I,
)


def _is_followup(text: str) -> bool:
    """True = уточнение предыдущего запроса (контекст уместен)."""
    return bool(_FOLLOWUP_RE.search((text or "").lower()))

# Допустимые значения (Enum)
ACTIONS = {
    "play_music", "radio", "next", "previous", "pause", "resume", "stop",
    "search", "text", "clarify",
}
FRESHNESS = {"new", "recent", "any", "old_ok"}          # new=0-2г, recent=2-5л, any, old_ok=классика
POPULARITY = {"popular", "any", "niche"}
# ТЗ п.5: дополнительные поля интерпретатора (безопасные fallback-значения)
MODES = {"exact", "similar", "playlist", "radio"}        # желаемый режим исполнения
DISCOVERY = {"none", "low", "medium", "high"}            # открытость новому
DIVERSITY = {"low", "medium", "high"}                    # предпочтение разнообразия

# JSON-схема для response_format=json_schema (проверено: Xiaomi принимает)
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "query": {"type": ["string", "null"]},
        "mood": {"type": ["string", "null"]},
        "activity": {"type": ["string", "null"]},
        "energy": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "genres": {"type": "array", "items": {"type": "string"}},
        "exclude_genres": {"type": "array", "items": {"type": "string"}},
        "seed_artist": {"type": ["string", "null"]},
        "seed_artists": {"type": "array", "items": {"type": "string"}},
        "seed_track": {"type": ["string", "null"]},
        "exclude_artists": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": ["string", "null"], "enum": sorted(MODES)},
        "discovery_level": {"type": ["string", "null"], "enum": sorted(DISCOVERY)},
        "diversity_preference": {"type": ["string", "null"], "enum": sorted(DIVERSITY)},
        "clarification_required": {"type": ["boolean", "null"]},
        "freshness": {"type": ["string", "null"], "enum": sorted(FRESHNESS)},
        "decade": {"type": ["integer", "null"], "enum": [1970, 1980, 1990, 2000, 2010]},
        "popularity": {"type": ["string", "null"], "enum": sorted(POPULARITY)},
        "similarity": {"type": ["string", "null"], "enum": ["low", "medium", "high"]},
        "language": {"type": ["string", "null"]},
        "include_classics": {"type": "boolean"},
        "exclude_covers": {"type": "boolean"},
        "clarify_question": {"type": ["string", "null"]},
    },
    "required": [
        "action", "query", "mood", "activity", "energy", "genres",
        "exclude_genres", "seed_artist", "seed_artists", "seed_track",
        "exclude_artists", "mode", "discovery_level", "diversity_preference",
        "clarification_required",
        "freshness", "decade", "popularity", "similarity", "language",
        "include_classics", "exclude_covers", "clarify_question",
    ],
}

_SYSTEM = (
    "Ты — музыкальный парсер музыкальных просьб. Отвечай ТОЛЬКО валидным JSON "
    "по заданной схеме, без markdown и пояснений. "
    "action=search ДЛЯ ЛЮБОЙ музыкальной просьбы, даже если источник или "
    "точное название неясны: «песня из сериала/фильма/игры X», «саундтрек X», "
    "«музыка из X», «найди/включи/поставь X», «что-нибудь подобное» — ВСЕГДА "
    "search (уточнять НЕ нужно, запрос уточнится сам по результатам). "
    "action=clarify ТОЛЬКО если просьба совсем не про музыку и не содержит "
    "ни одного музыкального слова (пример: «который час»). "
    "Энергия: 1=спокойно ... 5=максимально бодро. "
    "freshness: new=последние 2 года, recent=2-5 лет, old_ok=классика/старое разрешено, "
    "any=не важно. exclude_covers=true если пользователь не просит каверы/ремиксы. "
    "exclude_artists — артистов, которых НЕЛЬЗЯ включать («без этого исполнителя»). "
    "mode: similar=похожее, playlist=подборка/микс, exact=точный исполнитель+название, "
    "radio=радио. discovery_level — насколько открывать новое (high если просит новое). "
    "diversity_preference — разнообразие артистов (high если жалуется на однообразие). "
    "decade УКАЗЫВАЙ только если пользователь явно написал («90-х», «2000-х»), "
    "иначе 0. НИЧЕГО не выдумывай: неизвестное поле = null/пусто."
)


def _decade_from_text(text: str) -> int:
    """Декада из текста просьбы («90-х» → 1990). 0 = не найдено.

    Используется и offline, и как подстраховка LLM-ветки (LLM может
    вернуть decade=null, хотя юзер явно написал «90-х»).
    """
    t = (text or "").lower().replace("ё", "е")
    for _pat, _dec in (
        (r"90[- ]?(х|е|лет)|девяност", 1990),
        (r"80[- ]?(х|е|лет)|восьмидесят", 1980),
        (r"70[- ]?(х|е|лет)|семидесят", 1970),
        (r"2000[- ]?(х|е|лет)|двухтысячн|нулевых", 2000),
        (r"2010[- ]?(х|е|лет)", 2010),
    ):
        if re.search(_pat, t):
            return _dec
    return 0


@dataclass
class MusicIntent:
    """Структурированный музыкальный intent (один разбор просьбы)."""
    action: str = "search"                    # play_music|radio|search|text|clarify|...
    query: str = ""                           # поисковая строка (LLM может дать готовую)
    mood: str = ""                            # настроение (energetic, calm, ...)
    activity: str = ""                        # ситуация (driving, workout, sleep, ...)
    energy: int = 0                           # 1..5 (0 = не указано)
    genres: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    seed_artist: str = ""                     # артист-ориентир (один)
    seed_artists: list[str] = field(default_factory=list)  # ТЗ 13: список якорей
    seed_track: str = ""                      # трек-ориентир
    freshness: str = "any"                    # new|recent|any|old_ok
    decade: int = 0                           # 1970|1980|1990|2000|2010 (0 = не указано)
    popularity: str = "any"                   # popular|any|niche
    similarity: str = "medium"                # low|medium|high
    language: str = ""                        # ru|en|...
    include_classics: bool = False            # разрешить старые треки
    exclude_covers: bool = True               # исключить каверы по умолчанию
    clarify_question: str = ""                # если action=clarify
    # ТЗ п.5: расширенная схема интерпретатора
    mode: str = ""                            # exact|similar|playlist|radio ("" = решает сервер)
    exclude_artists: list[str] = field(default_factory=list)   # «без этого артиста»
    discovery_level: str = ""                 # none|low|medium|high — открытие нового
    diversity_preference: str = ""            # low|medium|high — разнообразие артистов
    clarification_required: bool = False      # action=clarify или неясная просьба
    raw: str = ""                             # сырой JSON (для логов)

    # ── производные ──────────────────────────────────────────────────────
    def to_search_queries(self) -> list[str]:
        """Несколько поисковых строк из одного intent (для multi-strategy сбора).

        Порядок: точный запрос LLM → mood+activity+genres → seed_artist.
        Дедуплицирует и убирает пустые.
        """
        out: list[str] = []

        def add(s: str) -> None:
            s = (s or "").strip()
            if s and len(s) >= 2 and s not in out:
                out.append(s)

        # 1. Готовый запрос от LLM (самый точный)
        add(self.query)
        # 2. mood + activity + genres (для подборок)
        parts = []
        if self.mood:
            parts.append(self.mood)
        if self.activity and self.activity not in self.mood:
            parts.append(self.activity)
        if self.genres:
            parts.append(" ".join(self.genres[:2]))
        if parts:
            add(" ".join(parts))
        # 3. seed_artist (отдельный — для артист-якоря)
        if self.seed_artist:
            add(self.seed_artist)
            if self.genres:
                add(f"{self.seed_artist} {' '.join(self.genres[:1])}")
        # 4. Фолбэк: только жанры
        if not out and self.genres:
            add(" ".join(self.genres[:2]))
        return out

    def to_rec_kwargs(self) -> dict:
        """Параметры для recommender.rank()."""
        mode = "exact"
        if self.action in ("radio",):
            mode = "artist_radio"
        elif self.mood or self.activity or self.genres or self.energy:
            mode = "mood"
        elif self.popularity == "popular":
            mode = "popular"
        freshness = self.freshness or "any"
        if self.include_classics:
            freshness = "old_ok"
        return {
            "mode": mode,
            "freshness": freshness,
            "genres": list(self.genres),
            "seed_artist": self.seed_artist,
        }

    def to_dict(self) -> dict:
        """Сериализация (для логов/API)."""
        return {
            "action": self.action, "query": self.query, "mood": self.mood,
            "activity": self.activity, "energy": self.energy,
            "genres": self.genres, "exclude_genres": self.exclude_genres,
            "seed_artist": self.seed_artist, "seed_artists": self.seed_artists,
            "seed_track": self.seed_track,
            "freshness": self.freshness, "decade": self.decade,
            "popularity": self.popularity,
            "similarity": self.similarity, "language": self.language,
            "include_classics": self.include_classics,
            "exclude_covers": self.exclude_covers,
            "clarify_question": self.clarify_question,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MusicIntent":
        """Создание из dict с санитацией значений."""
        def _s(v, default=""):
            return (v if isinstance(v, str) else default) or default

        def _list(v):
            if isinstance(v, list):
                return [str(x) for x in v if x][:6]
            return []

        def _bool(v):
            return bool(v) if isinstance(v, bool) else False

        action = _s(d.get("action"), "search")
        if action not in ACTIONS:
            action = "search"
        freshness = _s(d.get("freshness"), "any")
        if freshness not in FRESHNESS:
            freshness = "any"
        popularity = _s(d.get("popularity"), "any")
        if popularity not in POPULARITY:
            popularity = "any"
        try:
            energy = int(d.get("energy") or 0)
        except (TypeError, ValueError):
            energy = 0
        if not (0 <= energy <= 5):
            energy = 0
        # декада запроса («90-х» → 1990) — только разрешённые десятилетия
        try:
            decade = int(d.get("decade") or 0)
        except (TypeError, ValueError):
            decade = 0
        if decade not in (1970, 1980, 1990, 2000, 2010):
            decade = 0
        # seed_artists: список якорей; seed_artist — обратная совместимость
        seed_artists = _list(d.get("seed_artists"))
        seed_artist = _s(d.get("seed_artist"))[:80]
        if not seed_artist and seed_artists:
            seed_artist = seed_artists[0]
        if seed_artist and seed_artist not in seed_artists:
            seed_artists.insert(0, seed_artist)
        return cls(
            action=action,
            query=_s(d.get("query"))[:200],
            mood=_s(d.get("mood"))[:60],
            activity=_s(d.get("activity"))[:60],
            energy=energy,
            genres=_list(d.get("genres")),
            exclude_genres=_list(d.get("exclude_genres")),
            seed_artist=seed_artist,
            seed_artists=seed_artists[:6],
            seed_track=_s(d.get("seed_track"))[:80],
            freshness=freshness,
            decade=decade,
            popularity=popularity,
            similarity=_s(d.get("similarity"), "medium") or "medium",
            language=_s(d.get("language"))[:10],
            include_classics=_bool(d.get("include_classics")),
            exclude_covers=_bool(d.get("exclude_covers", True)),
            clarify_question=_s(d.get("clarify_question"))[:200],
            # ТЗ п.5: новые поля с безопасным fallback (неизвестное → default)
            mode=(lambda m: m if m in MODES else "")(_s(d.get("mode"))),
            exclude_artists=[x[:60] for x in _list(d.get("exclude_artists"))],
            discovery_level=(lambda v: v if v in DISCOVERY else "")(_s(d.get("discovery_level"))),
            diversity_preference=(lambda v: v if v in DIVERSITY else "")(_s(d.get("diversity_preference"))),
            clarification_required=bool(d.get("clarification_required"))
            if isinstance(d.get("clarification_required"), bool)
            else (action == "clarify"),
        )


def _extract_json(text: str) -> dict | None:
    """Достаёт JSON из ответа LLM (на случай markdown-обёртки)."""
    if not text:
        return None
    t = text.strip()
    # убираем markdown-блоки
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # пробуем целиком
    try:
        d = json.loads(t)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    # ищем первый {...}
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            return None
    return None


async def parse_intent_llm(
    text: str,
    current_track: str = "",
    *,
    retry: bool = True,
) -> MusicIntent | None:
    """Разбирает просьбу в структурированный MusicIntent через LLM.

    Схема: response_format=json_schema (проверено). При ошибке — 1 retry с
    json_object. При повторной ошибке/недоступности → None (fallback на старый
    mood_mix_query).

    НИКОГДА не бросает исключение — LLM-ошибка не должна ронять обработку.
    """
    from .llm import _chat, _llm_ready, SYSTEM_PROMPT

    if not _llm_ready():
        return None

    base = (text or "").strip()
    if not base:
        return None
    # ТЗ п.10.11: кэш LLM-разборов — одинаковый запрос+контекст не идёт в LLM
    # повторно (дубли поисковых запросов = лишний расход и задержка).
    _ckey = f"{base.lower()}|{current_track.strip().lower()}"
    _hit = _INTENT_CACHE.get(_ckey)
    if _hit is not None:
        logger.debug("music_intent cache hit: %r", base[:40])
        try:
            from . import metrics as _mt
            _mt.event("intent_cache_hit")
        except Exception:  # noqa: BLE001
            pass
        _cached = _copy.deepcopy(_hit)
        # контекст обязан обновиться и при кэш-хите (иначе уточнения
        # унаследуют УСТАРЕВШИЙ intent от промежуточного запроса)
        global _LAST_INTENT
        _LAST_INTENT = {
            "text": base[:80],
            "action": _cached.action,
            "mood": _cached.mood,
            "genres": _cached.genres[:4],
            "seed_artists": _cached.seed_artists[:3],
        }
        return _cached

    ctx = f"Сейчас играет: {current_track}" if current_track else "Трек ещё не выбран."
    # ТЗ п.4: контекст прошлой команды — ТОЛЬКО для уточнений («ещё похожее»,
    # «поспокойнее»); независимый запрос не наследует seed/decade прошлого.
    prev = _prev_context() if _is_followup(base) else ""
    user_prompt = (
        f"Просьба пользователя: {base}\n{ctx}\n"
        + (f"{prev}\n" if prev else "")
        + "Определи музыкальный intent. Верни ТОЛЬКО JSON по схеме."
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    # Попытка 1: json_schema (строгая)
    rf_schema = {"type": "json_schema", "json_schema": {"name": "music_intent", "strict": True, "schema": INTENT_SCHEMA}}
    msg = await _chat(messages, max_tokens=700, temperature=0.1, response_format=rf_schema)
    content = (msg.get("content") or "").strip() if msg else ""
    data = _extract_json(content) if content else None

    # Попытка 2 (retry): json_object (мягче)
    if data is None and retry:
        logger.info("music_intent schema-parse failed, retry json_object for %r", base[:60])
        rf_obj = {"type": "json_object"}
        msg = await _chat(messages, max_tokens=700, temperature=0.1, response_format=rf_obj)
        content = (msg.get("content") or "").strip() if msg else ""
        data = _extract_json(content) if content else None

    if not data:
        logger.warning("music_intent failed for %r (нет JSON)", base[:60])
        try:
            from . import metrics as _mt_n
            _mt_n.event("intent_parse_failed")
        except Exception:  # noqa: BLE001
            pass
        return None

    try:
        intent = MusicIntent.from_dict(data)
        intent.raw = json.dumps(data, ensure_ascii=False)[:500]
        # Валидация обязательных полей (ТЗ 4.1): action должен быть осмысленным
        if not intent.action:
            intent.action = "search"
        # Декада: ТОЛЬКО из явных маркеров исходного текста («90-х», «2000-х»).
        # LLM выдумывает декады из ниоткуда (ТЗ «не выдумывать характеристики»):
        # «танцевальная подборка» → LLM decade=2010/1990 → якорь «1990-х»
        # в поиске уводил в старые танцы (журнал 10:51).
        intent.decade = _decade_from_text(base)
        # запоминаем контекст для СЛЕДУЮЩЕЙ команды (ТЗ п.13)
        # global _LAST_INTENT объявлен выше (кэш-ветка) — дубль запрещён
        _LAST_INTENT = {
            "text": base[:80],
            "action": intent.action,
            "mood": intent.mood,
            "genres": intent.genres[:4],
            "seed_artists": intent.seed_artists[:3],
        }
        logger.info(
            "music_intent: action=%s mood=%r activity=%r energy=%s genres=%s fresh=%s seed=%r",
            intent.action, intent.mood, intent.activity, intent.energy,
            intent.genres, intent.freshness, intent.seed_artist,
        )
        # ТЗ п.10.11: кладём в кэш (вытеснение старейшего при переполнении)
        if _ckey in _INTENT_CACHE:
            _INTENT_CACHE.pop(_ckey, None)
        elif len(_INTENT_CACHE) >= 32:
            _INTENT_CACHE.pop(next(iter(_INTENT_CACHE)), None)
        _INTENT_CACHE[_ckey] = _copy.deepcopy(intent)
        return intent
    except Exception as e:  # noqa: BLE001
        logger.warning("music_intent parse error: %s", e)
        return None


# ─── Быстрый фолбэк без LLM (детерминированный) ──────────────────────────
_KEYWORDS_MOOD = {
    "бодр": "energetic", "весел": "happy", "весёл": "happy",
    "спок": "calm", "груст": "sad", "меланхол": "melancholic",
    "энерг": "energetic", "расслаб": "relaxed", "спокойн": "calm",
    "тревож": "anxious", "радост": "happy", "ностальг": "nostalgic",
}
_KEYWORDS_ACTIVITY = {
    "дорог": "driving", "трениров": "workout", "зал": "gym",
    "работ": "work", "сон": "sleep", "сна": "sleep", "вечерин": "party",
    "утр": "morning", "вечер": "evening", "ноч": "night",
    "подъезд": "commute", "беж": "running",
}


def parse_intent_offline(text: str) -> MusicIntent | None:
    """Детерминированный разбор БЕЗ LLM (fallback).

    Определяет mood/activity по ключевым словам — быстро и без сети.
    Для точных запросов («включи X») возвращает action=play_music.
    """
    if not text:
        return None
    t = text.strip().lower().replace("ё", "е")
    if not t:
        return None

    intent = MusicIntent()
    # Ключевые слова mood/activity
    for kw, mood in _KEYWORDS_MOOD.items():
        if kw in t:
            intent.mood = mood
            break
    for kw, act in _KEYWORDS_ACTIVITY.items():
        if kw in t:
            intent.activity = act
            break
    # Жанры (простой словарь)
    genre_map = {
        "рэп": "русский рэп", "хип-хоп": "хип-хоп", "хип хоп": "хип-хоп",
        "рок": "рок", "поп": "поп", "джаз": "джаз", "техно": "техно",
        "хаус": "хаус", "электрон": "электронная музыка", "релакс": "релакс",
        "шансон": "шансон", "классик": "классика", "инструментал": "инструментал",
        "лирическ": "лирические песни", "колыбельн": "колыбельные",
    }
    for kw, g in genre_map.items():
        if kw in t and g not in intent.genres:
            intent.genres.append(g)
    # Артист (последнее слово-имя) — с фильтром стоп-слов: «классической русской»
    # не должен становиться seed-артистом (патч 2: мусорный seed ломает поиск).
    from .recommender import _extract_artist_from_query
    intent.seed_artist = _extract_artist_from_query(t)
    if intent.seed_artist and any(
        w in intent.seed_artist
        for w in (
            "классическ", "старин", "девяност", "русск", "зарубежн",
            "современ", "энергич", "спокойн", "вечерн", "утренн", "дорожн",
        )
    ):
        intent.seed_artist = ""
    # action
    if intent.mood or intent.activity or intent.genres:
        intent.action = "play_music"
        intent.query = text
    elif re.search(r"(включи|поставь|найди|запусти|поиграй)", t):
        intent.action = "play_music"
        intent.query = text
    else:
        intent.action = "search"
        intent.query = text
    # freshness: «старин/классик/90-х/2000-х» → old_ok; «нов/поновее/свеж» → recent
    if re.search(r"(старин|классик|90-х|2000-х|советск|ретро)", t):
        intent.freshness = "old_ok"
        intent.include_classics = True
    elif re.search(r"(нов|поновее|свеж|последн)", t):
        intent.freshness = "recent"
    # Декада запроса (п.4 ТЗ: ограничения по свежести) — «90-х» → 1990.
    # Используется ранжером: классика сортируется по БЛИЗОСТИ к декаде.
    intent.decade = _decade_from_text(t)
    if intent.seed_artist or intent.mood or intent.genres:
        logger.info("music_intent offline: mood=%r activity=%r genres=%s seed=%r",
                    intent.mood, intent.activity, intent.genres, intent.seed_artist)
        return intent
    return None
