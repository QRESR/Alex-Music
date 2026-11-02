"""Рекомендательный движок (P1 ТЗ) — отдельный патч с feature-флагом.

ВАЖНО: модуль НЕ трогает существующую логику, пока config.USE_NEW_RANKER=False.
При False старый _rank_by_artist() продолжает работать как раньше.

Что делает recommender.rank():
  1. Универсальное скоринг (веса в config/RECOMMENDER_WEIGHTS)
  2. Гибкая свежесть (новинки/актуальные/классика — НЕ жёсткий фильтр)
  3. Re-ranking на разнообразие (≤ max_per_artist треков одного артиста)
  4. Метрики в лог (artist_match@top5, freshness, elapsed)

Веса зависят от ТИПА запроса (по config.QUERY_TYPES):
  exact        — точный поиск: артист+название максимум
  mood         — подборка по настроению: жанр+энергия+настроение
  artist_radio — радио от артиста: музыкальная похожесть
  discovery    — открытия: разнообразие+новизна
  popular      — хиты: популярность

Использование (после feature-флага):
    from recommender import rank, metrics_snapshot
    tracks = rank(tracks, query=..., mode="mood", freshness=...)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("recommender")

# ─── СТАНДАРТНЫЕ ВЕСА (веса ТЗ, разложены по типам запросов) ─────────────
# Каждый тип запроса — своя сумма ~100. Это СТАРТОВЫЕ коэффициенты,
# корректируются в config.RECOMMENDER_WEIGHTS.
DEFAULT_WEIGHTS: dict[str, dict[str, int]] = {
    "exact": {
        "artist_match": 40,    # точное совпадение артиста — ВАЖНЕЕ ВСЕГО
        # ТЗ п.13.3/п.7 «Лучший ответ первым»: title_match 25→8 — иначе
        # «BASTA! - Basta» (чужая группа, title=баста) обгоняет НАСТОЯЩУЮ
        # Басту: 40+25+10=75 против 40+0+20=60 (живой журнал 11:56).
        "title_match": 8,
        "genre_fit": 5,
        "popularity": 5,
        "freshness": 5,
        "taste": 20,
        "profile_genre": 8,
        "audio_sim": 3,        # ТЗ п.7: CLAP — дополнительный признак, НЕ порядок
    },
    "mood": {
        "artist_match": 10,
        # П.1 (2026-11-01): НЕ по названию песни — title_match 5→0. Отбор по
        # жанру/свежести/вкусу/звучанию (audio_sim), имя артиста — слабый признак.
        "title_match": 0,
        "genre_fit": 30,
        "popularity": 10,
        "freshness": 25,
        "taste": 20,
        "profile_genre": 14,
        "audio_sim": 18,       # по звучанию здесь главное — но НЕ определяет порядок
    },
    "artist_radio": {
        "artist_match": 35,
        "title_match": 10,
        "genre_fit": 25,
        "popularity": 5,
        "freshness": 5,
        "taste": 20,
        "profile_genre": 10,
        "audio_sim": 12,
    },
    "discovery": {
        "artist_match": 5,
        "title_match": 5,
        "genre_fit": 20,
        "popularity": 5,
        "freshness": 35,
        "taste": 30,
        "profile_genre": 12,
        "audio_sim": 20,       # открытия: аудио-похожесть к контексту важна
    },
    "popular": {
        "artist_match": 20,
        "title_match": 10,
        "genre_fit": 10,
        "popularity": 40,
        "freshness": 5,
        "taste": 15,
        "profile_genre": 5,
        "audio_sim": 8,
    },
    # ТЗ similar п.5: похожие от исходного трека. CLAP-сходствие — крупный,
    # но НЕ единственный признак (сумма признаков ~108, audio_sim ≤23%):
    # вкус/жанр/свежесть/популярность решают, «звучит ли это как продолжение».
    "similar": {
        "artist_match": 12,    # исходный артист допустим (1-2 из 20 через MMR)
        # П.1 (2026-11-01): НЕ по названию — title_match 4→0. Порядок решают
        # audio_sim (CLAP по звучанию) + вкус + жанр, не совпадение названия.
        "title_match": 0,
        "genre_fit": 25,
        "popularity": 8,
        "freshness": 12,
        "taste": 22,
        "profile_genre": 10,
        "audio_sim": 25,       # CLAP audio-to-audio — главный ОТДЕЛЬНЫЙ признак
    },
}


@dataclass
class RankContext:
    """Всё, что нужно движку для скоринга одного трека."""
    query: str = ""                   # исходный запрос
    seed_artist: str = ""             # артист-якорь (из ctx/note/intent)
    mode: str = "exact"               # exact|mood|artist_radio|discovery|popular
    freshness: str = "any"            # new|recent|any|old_ok
    genres: list[str] = field(default_factory=list)
    exclude_artists: list[str] = field(default_factory=list)
    banned_track_ids: set[str] = field(default_factory=set)
    banned_artists: set[str] = field(default_factory=set)
    liked_artists: set[str] = field(default_factory=set)   # lowercase
    disliked_artists: set[str] = field(default_factory=set)
    # P1 ТЗ 10.3: защита от однообразия — недавно сыгранные id/артисты.
    # Повтор в недавней истории → штраф (но НЕ жёсткое удаление — точный
    # поиск может намеренно вернуть уже сыгранный трек).
    recent_track_ids: set[str] = field(default_factory=set)    # сыграно за 7-14 дней
    recent_artists: dict[str, int] = field(default_factory=dict)  # artist -> count за период
    # Патч 2 шаг 15 (ТЗ п.8): алиасы якоря из MusicBrainz
    # («Miyagi» ↔ «Мияги», «Баста» ↔ «Basta») — расширяют artist_match.
    seed_aliases: set[str] = field(default_factory=set)   # нормализованные
    # ТЗ п.4/п.7 (патч 2): CLAP-похожесть audio→reference как ОДИН из признаков.
    # Заполняется ОТДЕЛЬНЫМ механизмом (embeddings) ДО ранжера, живёт в
    # RankContext.audio_sim: track_id -> 0..1. Ранжер СКЛАДЫВАЕТ её с
    # вкусом/жанром/свежестью — она НЕ определяет итоговый порядок сама.
    audio_sim: dict[str, float] = field(default_factory=dict)
    # ТЗ п.3: веса артистов из истории обратной связи (с затуханием).
    # artist (lowercase) -> вес [-..+]. Бинарный liked/disliked приоритетнее;
    # вес даёт градацию «любимый vs слушал однажды vs надоел».
    artist_weights: dict[str, float] = field(default_factory=dict)
    # ТЗ п.3.2/п.14: предпочитаемые ЖАНРЫ из истории (genre_scores с
    # затуханием) — «любимый/нежелательный» genre как признак ранжирования.
    # genre (lowercase) -> вес [-..+]; нейтраль при отсутствии данных.
    genre_weights: dict[str, float] = field(default_factory=dict)
    # ТЗ п.6 «допустимые отклонения от исходного артиста»: high — держим тему
    # якоря, low — можно уходить в другие стили. Множитель artist_match.
    similarity: str = "medium"                 # low|medium|high
    weights: dict = field(default_factory=dict)            # override из config


# ─── Нормализация ──────────────────────────────────────────────────────────
_NON_ARTIST_WORDS = {
    "включи", "включить", "поставь", "поставить", "поиграй", "поиграть",
    "найди", "найти", "запусти", "запустить", "хочу", "хочется", "давай",
    "мне", "нам", "нужно", "надо", "please", "play", "the", "and",
    "трек", "треки", "песня", "песни", "песню", "музыка", "музыку",
    "исполнитель", "исполнитель", "исполнитель", "artist", "singer",
    "подборку", "подборка", "радио", "radio", "микс", "mix",
    # P1: служебные слова из MusicIntent/LLM, которые НЕ артисты
    # (ловили в логах: seed='вечером', 'послушать', 'саунд', 'дороги')
    "вечером", "утром", "ночью", "днем", "днём", "дороги", "дорога",
    "работы", "работа", "тренировки", "тренировка", "стола", "улице",
    "настроения", "послушать", "поиграть", "саунд", "саундтрек",
    "подборки", "подборок", "звучит", "звучать", "поставить",
    # ФИКС ТЗ «очень странные дела» (журнал 12:41): слова-характеристики и
    # контекста НЕ артисты → не должны быть якорем («очень» + «хиты» = мусор
    # Яндекса). _NON_ARTIST_WORDS — set, дублей не бывает.
    "очень", "какой", "какая", "какие", "чтоб", "чтобы", "просто", "совсем",
    # Бенч 2026-10-31: якорь «годов» на запросе «музыка 90-х годов»
    # (decade-anchor «музыка годов 1990-х») → seed='годов' → мусорный
    # match@5=0.00. Год — не артист.
    "годов", "годам", "годами", "годах",
    # Бенч 2026-10-31: «музыка на 8 марта» → seed='марта' → транслит
    # toks={'marta'} → artist_hits=6 (испанки Marta/Marta Zuñiga/Caruso-Marta)
    # → в топе «orquestas suenate los mocos». Даты — не артисты.
    "марта", "март", "марте", "мартов", "января", "февраля", "апреля",
    "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря",
    "фильм", "фильма", "фильме", "сериала", "сериале", "сериалов",
    "саундтрека", "саундтреки", "трека", "треке", "петь", "спеть",
    # ФИКС ЖАЛОБ 2026-11-01 (живые запросы): слепой fallback брал ПОСЛЕДНЕЕ
    # слово как seed, а оно — не артист:
    #   «Зомби remastered» → seed='remastered' (0 artist_hits) → отбросил все
    #   Zombie/Зомби-треки, title-fallback оставил 3 мусорных (Voluminous Swath/
    #   DigitalDreamer/Echomen «Remastered»); ВК с тем же якорем обрезал 10
    #   Cranberries до 1. «песни из очень странные дела» → seed='дела' → рэп
    #   «Очень странные дела» вместо OST. «зомберы мастерят» → seed='мастерят'.
    # Ремастер/мастеринг/характеристики — НЕ имя артиста (по аналогии с
    # «годов»/«марта» из бенча 31 окт).
    "remastered", "remaster", "remastering", "mastered", "mastering",
    "издание", "версия", "версии", "версию", "версией", "качество",
    "дело", "дела", "дел", "делах", "делами",
    "мастерят", "мастерить", "мастерит", "мастерю",
    "странные", "странный", "странных", "оченьстранные",
}

_SUFFIXES = (
    "ский", "цкий", "ова", "ева", "ина", "ына", "ова", "ева",
    "ович", "евич", "ович", "евич", "льный", "тельный", "льный",
)


def _norm(s: str) -> str:
    """Нормализация строки для сравнения (нижний регистр, ё→е, без знаков)."""
    if not s:
        return ""
    s = s.lower().replace("ё", "е")
    # транслит в латиницу для универсального сравнения
    try:
        from .ym_client import translit_to_latin
        s = translit_to_latin(s)
    except Exception:  # noqa: BLE001
        pass
    return re.sub(r"[^0-9a-zа-я]+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    """Значимые слова (≥3 символов)."""
    return {t for t in _norm(s).split() if len(t) >= 3}


def _extract_artist_from_query(query: str) -> str:
    """Достаёт артист из запроса («включи баста» → «баста»).

    Эвристика: берём ПОСЛЕДНЕЕ слово-имя (не жанр/служебное), как в llm._extract_artist.
    """
    if not query:
        return ""
    words = re.split(r"[^0-9A-Za-zА-ЯЁа-яё]+", query.strip())
    candidates = []
    for w in words:
        if len(w) < 3:
            continue
        lw = w.lower().replace("ё", "е")
        if lw in _NON_ARTIST_WORDS:
            continue
        # жанры/ситуации/числа подстрокой
        if any(g in lw for g in (
            "поп", "рэп", "рок", "джаз", "музык", "песн", "хит", "ремикс",
            "подбор", "танц", "спок", "весел", "весёл", "груст", "бодр",
            "электр", "зарубеж", "русск", "современ", "для", "тренир",
            "включи", "постав", "поигра", "запусти", "найди",
        )):
            continue
        if len(lw) >= 5 and lw.endswith(_SUFFIXES):
            continue
        if lw.isdigit():
            continue
        candidates.append(w)
    return candidates[-1] if candidates else ""


def query_artist_candidates(query: str) -> list[str]:
    """ВСЕ слова-кандидаты на артиста из запроса, в порядке встречаемости.

    «асти модный поп» → ['Асти', 'модный'] (жанр/стоп отброшены)."""
    if not query:
        return []
    words = re.split(r"[^0-9A-Za-zА-ЯЁа-яё]+", query.strip())
    out: list[str] = []
    for w in words:
        if len(w) < 3:
            continue
        lw = w.lower().replace("ё", "е")
        if lw in _NON_ARTIST_WORDS:
            continue
        if any(g in lw for g in (
            "поп", "рэп", "рок", "джаз", "музык", "песн", "хит", "ремикс",
            "подбор", "танц", "спок", "весел", "весёл", "груст", "бодр",
            "электр", "зарубеж", "русск", "современ", "для", "тренир",
            "включи", "постав", "поигра", "запусти", "найди",
        )):
            continue
        if len(lw) >= 5 and lw.endswith(_SUFFIXES):
            continue
        if lw.isdigit():
            continue
        out.append(w)
    return out


def artist_hits_for(token: str, tracks: list) -> int:
    """Сколько треков пула имеют токен в имени исполнителя (норм+транслит)."""
    if not token or not tracks:
        return 0
    tok = _norm(token)
    if not tok:
        return 0
    n = 0
    for t in tracks:
        art = _norm(t.get("artist", "") or "" if isinstance(t, dict)
                    else getattr(t, "artist", "") or "")
        if any(x in art for x in tok.split() if len(x) >= 3):
            n += 1
    return n


def best_seed_from_query(query: str, tracks: list) -> tuple[str, int]:
    """Якорь, ВЫБРАННЫЙ ПО РЕАЛЬНОМУ ПУЛУ, а не по эвристике слова.

    Проблема: _extract_artist_from_query берёт ПОСЛЕДНЕЕ слово →
    «асти модный поп» → 'модный', «баста осень» → 'осень'. Здесь каждый
    кандидат-токен запроса проверяется на реальные артист-хиты в уже
    собранном пуле и побеждает тот, у кого больше хитов (при равенстве —
    первый в запросе): «асти модный поп» → 'асти' (ARTIK & ASTI в топе).

    Возвращает (seed, hits). hits=0 → ('', 0) — не якорим слепым словом.
    """
    best, best_n = "", 0
    for cand in query_artist_candidates(query):
        n = artist_hits_for(cand, tracks)
        if n > best_n:
            best, best_n = cand, n
            if best_n >= 3:
                break  # достаточно, дальше не ищем
    return best, best_n


# ─── Скоринг ───────────────────────────────────────────────────────────────
def _score_track(t: dict, ctx: RankContext, w: dict) -> float:
    """Итоговый score трека (0..100 в шкале весов, НЕ нормализован)."""
    artist = t.get("artist", "") or ""
    title = t.get("title", "") or ""
    year = int(t.get("year") or 0)
    rank = int(t.get("rank") or 0)
    popularity_raw = float(t.get("popularity") or 0)  # может быть из API

    # 1. artist_match (0 или 1)
    artist_match = 0
    if ctx.seed_artist:
        a_norm = _norm(artist)
        s_norm = _norm(ctx.seed_artist)
        if s_norm and s_norm in a_norm:
            artist_match = 1
        else:
            # токен-совпадение
            a_toks = _tokens(artist)
            s_toks = _tokens(ctx.seed_artist)
            if a_toks and s_toks and (a_toks & s_toks):
                artist_match = 1
            # алиасы MusicBrainz (шаг 15): «Miyagi» ↔ «Мияги & Эндшпи»
            elif ctx.seed_aliases:
                al_norm = _norm(artist)
                if any(al and al in al_norm for al in ctx.seed_aliases):
                    artist_match = 1
                else:
                    al_toks = _tokens(artist)
                    if al_toks and any(al_toks & _tokens(a) for a in ctx.seed_aliases if a):
                        artist_match = 1
    elif ctx.query:
        # нет явного seed_artist — извлекаем из query
        qa = _norm(_extract_artist_from_query(ctx.query))
        if qa and qa in _norm(artist):
            artist_match = 1

    # 2. title_match (0..1)
    title_match = 0.0
    if ctx.query:
        q_toks = _tokens(ctx.query)
        t_toks = _tokens(title)
        if q_toks and t_toks:
            inter = q_toks & t_toks
            if inter:
                title_match = len(inter) / max(len(q_toks), 1)

    # 3. genre_fit (0..1)
    genre_fit = 0.0
    if ctx.genres:
        a_l = artist.lower()
        t_l = title.lower()
        for g in ctx.genres:
            g_l = g.lower()
            if g_l in a_l or g_l in t_l:
                genre_fit = 1.0
                break

    # 4. popularity (0..1) — rank позиция ИЛИ popularity_raw
    if rank > 0:
        popularity = max(0.0, 1.0 - rank / 20.0)  # rank 0-20 → 1.0-0.0
    elif popularity_raw > 0:
        popularity = min(1.0, popularity_raw)
    else:
        popularity = 0.5  # неизвестно — нейтрально

    # 5. freshness (0..1) — новинки выше, НО не 0 для классики
    freshness = 0.5  # дефолт для неизвестной даты
    if year > 0:
        current_year = time.gmtime(time.time()).tm_year
        age = current_year - year
        if ctx.freshness == "old_ok":
            freshness = 0.5  # запрос «классика/90-е» — свежесть не важна
        elif age <= 2:
            freshness = 1.0     # новинка (0-2 года)
        elif age <= 5:
            freshness = 0.7     # актуальная (2-5 лет)
        else:
            freshness = 0.3     # классика (>5 лет) — не ноль, но понижено

    # 6. taste (0..1)
    taste = 0.5  # нейтрально
    a_low = artist.lower().strip()
    if a_low:
        if a_low in ctx.liked_artists:
            taste = 1.0
        elif a_low in ctx.disliked_artists:
            taste = 0.0
        else:
            # ТЗ п.3: вес из истории (с затуханием) — градация вместо
            # плоского «0.5 всем»: лайки/дослушивания поднимают (до ~0.85
            # за лайк), скипы/дизлайки опускают (0.0 — при -3 и ниже).
            # ВАЖНО: имя _aw, НЕ w — w уже занята весами признаков!
            _aw = 0.0
            aw = getattr(ctx, "artist_weights", None) or {}
            if aw:
                _aw = float(aw.get(a_low, 0.0))
                if _aw == 0.0:
                    # «Баста, Моя Мишель» → ключ «баста»: подстрока в обе стороны
                    for k, kv in aw.items():
                        if len(k) >= 4 and (k in a_low or a_low in k):
                            _aw = float(kv)
                            break
            if _aw:
                taste = max(0.0, min(1.0, 0.5 + _aw / 6.0))

    # Итоговый score (взвешенная сумма)
    # ТЗ п.4/п.7: audio_sim (CLAP) — ОДИН из признаков, а не ключ сортировки:
    # складывается с вкусом/жанром/свежестью/популярностью. Если для трека нет
    # эмбеддинга (кэш пуст) — 0, остальные признаки работают как прежде.

    # 6b. ТЗ п.3.2: ПРОФИЛЬНЫЕ жанры (genre_scores с затуханием 30 дней).
    # Нейтраль 0.5 без данных (нет профиля/нет жанров у трека) — признак НЕ
    # выдумывает, а усиливает уже заработанные лайками направления. Как и
    # taste: градация 0.5 + w/6 (лайк-жанр ~0.8, надоевший ~0.1-0.2).
    profile_genres = 0.5
    _pgw = getattr(ctx, "genre_weights", None) or {}
    if _pgw:
        _tg = [str(g).lower().strip() for g in (t.get("genres") or [])]
        _best_g = 0.0
        for _g in _tg:
            _gv = _pgw.get(_g)
            if _gv is None:
                # подстрочное: «rusrap» ↔ «russian rap», «synthwave» ↔ «synth»
                for _k, _kv in _pgw.items():
                    if len(_k) >= 4 and (_k in _g or _g in _k):
                        _gv = _kv
                        break
            if _gv is not None and abs(float(_gv)) > abs(_best_g):
                _best_g = float(_gv)
        if _best_g:
            profile_genres = max(0.0, min(1.0, 0.5 + _best_g / 6.0))

    audio_sim = 0.0
    if getattr(ctx, "audio_sim", None):
        tid_a = (t.get("id") or "") if isinstance(t, dict) else ""
        audio_sim = float(ctx.audio_sim.get(tid_a, 0.0))
        audio_sim = max(0.0, min(1.0, audio_sim))
    score = (
        w.get("artist_match", 0) * artist_match
        + w.get("title_match", 0) * title_match
        + w.get("genre_fit", 0) * genre_fit
        + w.get("popularity", 0) * popularity
        + w.get("freshness", 0) * freshness
        + w.get("taste", 0) * taste
        + w.get("audio_sim", 0) * audio_sim
        + w.get("profile_genre", 0) * profile_genres   # ТЗ п.3.2: жанры профиля
    )

    # P1 ТЗ 10.3: защита от однообразия — штраф за недавние повторы.
    # НЕ удаляем (точный поиск может вернуть уже сыгранный трек),
    # но понижаем score, чтобы свежее всплывало выше.
    tid = (t.get("id") or "") if isinstance(t, dict) else ""
    a_low = artist.lower().strip()
    if ctx.recent_track_ids and tid and tid in ctx.recent_track_ids:
        score -= 15   # уже играли недавно → ниже
    if ctx.recent_artists and a_low:
        cnt = ctx.recent_artists.get(a_low, 0)
        if cnt >= 3:
            score -= 20   # часто играли этого артиста → заметно ниже
        elif cnt >= 1:
            score -= 5    # играли → слегка ниже
    return score


# ─── Re-ranking для разнообразия ──────────────────────────────────────────
def _diversify(tracks: list[dict], max_per_artist: int = 3) -> list[dict]:
    """Ограничивает количество треков одного артиста подряд (≤ max_per_artist).

    Жадный алгоритм: проходим по отсортированному списку, если артист уже
    встречался max_per_artist раз — откладываем трек в резерв и добираем
    из резерва в конце.
    """
    if max_per_artist <= 0:
        return tracks

    def get_artist(t) -> str:
        if isinstance(t, dict):
            return (t.get("artist", "") or "").strip().lower()
        return (getattr(t, "artist", "") or "").strip().lower()

    result: list = []
    reserve: list = []
    artist_count: dict[str, int] = {}
    for t in tracks:
        a = get_artist(t)
        c = artist_count.get(a, 0)
        if c < max_per_artist:
            result.append(t)
            artist_count[a] = c + 1
        else:
            reserve.append(t)
    # резерв — в конец (но не более лимита)
    result.extend(reserve)
    return result


# ─── Метрики ───────────────────────────────────────────────────────────────
_metrics_log: list[dict] = []


def _log_metrics(query: str, tracks: list[dict], ctx: RankContext, elapsed: float) -> None:
    """Собирает метрики выдачи (по ТЗ 17.2)."""
    if not tracks:
        return
    # artist_match@top5
    top5 = tracks[:5]
    artist_matches = 0
    for t in top5:
        a = _norm(t.get("artist", ""))
        seed = _norm(ctx.seed_artist) or _norm(_extract_artist_from_query(ctx.query))
        if seed and seed in a:
            artist_matches += 1
    artist_match_rate = artist_matches / len(top5)

    # fresh count (год ≥ current_year - 2)
    cy = time.gmtime(time.time()).tm_year
    fresh_count = sum(1 for t in tracks if int(t.get("year") or 0) >= cy - 2)

    # дубликаты артистов
    seen_artists: dict[str, int] = {}
    for t in tracks:
        a = (t.get("artist") or "").strip().lower()
        seen_artists[a] = seen_artists.get(a, 0) + 1
    max_artist_repeats = max(seen_artists.values()) if seen_artists else 0

    m = {
        "query": query[:80],
        "mode": ctx.mode,
        "n_tracks": len(tracks),
        "artist_match@top5": round(artist_match_rate, 3),
        "fresh_count": fresh_count,
        "max_artist_repeats": max_artist_repeats,
        "elapsed_ms": int(elapsed * 1000),
        "ts": int(time.time()),
    }
    _metrics_log.append(m)
    # ограничиваем лог 500 записей
    if len(_metrics_log) > 500:
        _metrics_log[:] = _metrics_log[-500:]
    logger.info(
        "metrics: q=%r mode=%s n=%d match@5=%.2f fresh=%d maxrep=%d %dms",
        query[:50], ctx.mode, len(tracks), artist_match_rate,
        fresh_count, max_artist_repeats, m["elapsed_ms"],
    )


def metrics_snapshot(limit: int = 50) -> list[dict]:
    """Последние N метрик (для /api/metrics)."""
    return list(reversed(_metrics_log[-limit:]))


def metrics_reset() -> None:
    _metrics_log.clear()


# ─── Главная функция ───────────────────────────────────────────────────────
def rank(
    tracks: list,
    query: str = "",
    seed_artist: str = "",
    mode: str = "exact",
    freshness: str = "any",
    genres: list[str] | None = None,
    exclude_artists: list[str] | None = None,
    banned_track_ids: set[str] | None = None,
    banned_artists: set[str] | None = None,
    liked_artists: set[str] | None = None,
    disliked_artists: set[str] | None = None,
    recent_track_ids: set[str] | None = None,
    recent_artists: dict[str, int] | None = None,
    seed_aliases: set[str] | None = None,
    audio_sim: dict[str, float] | None = None,
    artist_weights: dict[str, float] | None = None,
    genre_weights: dict[str, float] | None = None,   # ТЗ п.3.2: жанры профиля
    similarity: str = "medium",               # ТЗ п.6: low|medium|high — отклонения от якоря
    weights: dict | None = None,
    max_per_artist: int = 3,
) -> list:
    """Ранжирует список треков (dict или Pydantic) через Recommendation Engine.

    audio_sim — ТЗ п.4/п.7: CLAP-похожесть track_id->0..1, посчитанная
    ОТДЕЛЬНЫМ механизмом. Только признак в взвешенной сумме, НЕ порядок.

    Возвращает тот же тип элементов, что и на вход (dict → dict, TrackMeta → TrackMeta).
    """
    if not tracks:
        return tracks

    t0 = time.time()

    # Определяем seed_artist если не задан
    if not seed_artist:
        seed_artist = _extract_artist_from_query(query)

    ctx = RankContext(
        query=query,
        seed_artist=seed_artist,
        mode=mode,
        freshness=freshness,
        genres=genres or [],
        exclude_artists=exclude_artists or [],
        banned_track_ids=banned_track_ids or set(),
        banned_artists=banned_artists or set(),
        liked_artists=liked_artists or set(),
        disliked_artists=disliked_artists or set(),
        recent_track_ids=recent_track_ids or set(),
        recent_artists=recent_artists or {},
        seed_aliases=seed_aliases or set(),
        audio_sim=audio_sim or {},
        artist_weights=artist_weights or {},
        genre_weights=genre_weights or {},
        similarity=(similarity if similarity in ("low", "medium", "high") else "medium"),
        weights=weights or {},
    )

    # Веса по режиму
    w = dict(DEFAULT_WEIGHTS.get(mode, DEFAULT_WEIGHTS["exact"]))
    w.update(ctx.weights)
    # ТЗ п.6 «допустимые отклонения»: similarity=high — ближе к якорю
    # (artist_match ×1.4), low — можно уходить в смежное (×0.7).
    # LLM не ранжирует — она только ЗАДАЁТ степень, множит бюджет движка.
    if ctx.similarity == "high":
        w["artist_match"] = int(w.get("artist_match", 0) * 1.4)
    elif ctx.similarity == "low":
        w["artist_match"] = int(w.get("artist_match", 0) * 0.7)

    # Фильтрация: ban-list
    filtered: list = []
    for t in tracks:
        tid = t.get("id") if isinstance(t, dict) else getattr(t, "id", "")
        art = (t.get("artist") if isinstance(t, dict) else getattr(t, "artist", "")) or ""
        a_low = art.strip().lower()
        if tid in ctx.banned_track_ids:
            continue
        if a_low in ctx.banned_artists:
            continue
        if ctx.exclude_artists and a_low in {a.lower().strip() for a in ctx.exclude_artists}:
            continue
        filtered.append(t)

    if not filtered:
        return filtered

    # Скоринг (для dict и Pydantic)
    def get_dict(t) -> dict:
        if isinstance(t, dict):
            return t
        # Pydantic
        try:
            return t.model_dump()
        except Exception:  # noqa: BLE001
            return {}

    scored: list[tuple[float, object]] = []
    for t in filtered:
        d = get_dict(t)
        s = _score_track(d, ctx, w)
        scored.append((s, t))

    # Сортировка по убыванию score (стабильная — сохраняет исходный порядок при равных)
    scored.sort(key=lambda x: -x[0])
    sorted_tracks = [t for _, t in scored]

    # Re-ranking: разнообразие (≤ max_per_artist треков одного артиста)
    if max_per_artist > 0:
        sorted_tracks = _diversify(sorted_tracks, max_per_artist)

    # EXPLORATION / DISCOVERY (ТЗ п.10, шаг 12 «Re-ranking и Discovery»):
    # «постепенно предлагать новых исполнителей, не в привычном кругу».
    # Если в первой половине рекомендательной выдачи ВСЕ артисты уже из
    # вкуса пользователя — поднимаем первого НЕзнакомного из глубины на
    # позицию ~60% списка (не топ — релевантность важнее, ТЗ п.10).
    # Точный поиск (mode=exact) НЕ трогаем: там юзер просил конкретное.
    if mode in ("mood", "artist_radio", "discovery") and ctx.liked_artists and len(sorted_tracks) >= 6:
        def _is_known(t) -> bool:
            a = (
                t.get("artist") if isinstance(t, dict) else getattr(t, "artist", "")
            ) or ""
            a = a.lower()
            return any(k and k in a for k in ctx.liked_artists)
        try:
            _head_n = max(5, len(sorted_tracks) // 2)
            _head = sorted_tracks[:_head_n]
            if _head and all(_is_known(t) for t in _head):
                for _i in range(_head_n, len(sorted_tracks)):
                    if not _is_known(sorted_tracks[_i]):
                        _moved = sorted_tracks.pop(_i)
                        _pos = max(3, len(sorted_tracks) * 3 // 5)
                        sorted_tracks.insert(_pos, _moved)
                        logger.info(
                            "exploration: новый артист %r поднят на позицию %d",
                            ((_moved.get("artist") if isinstance(_moved, dict) else getattr(_moved, "artist", "")) or "")[:30],
                            _pos,
                        )
                        break
        except Exception as e:  # noqa: BLE001
            logger.debug("exploration skipped: %s", e)

    # Метрики
    elapsed = time.time() - t0
    try:
        _log_metrics(query, [get_dict(t) for t in sorted_tracks], ctx, elapsed)
    except Exception as e:  # noqa: BLE001
        logger.debug("metrics log failed: %s", e)

    return sorted_tracks


# ---------------------------------------------------------------------------
# Diversity Reranking — ТЗ «Похожие» п.6 (обязательная часть):
# MMR (Maximal Marginal Relevance): баланс «подходит исходному» × «отличается
# от уже выбранных». Учитывает релевантность (суммарный score rank() ИЛИ
# порядок, если score не передан) и сходство метаданных между кандидатами.
# Гарантии (ТЗ п.6): не более per_artist_seed треков исходного исполнителя,
# не более per_artist_other — чужого; подряд один артист не повторяется без
# надобности (MMR сам оттесняет похожих). НЕ ухудшает rel ради разнообразия:
# lambda_ > 0.5 = релевантность важнее.
# ---------------------------------------------------------------------------
def _as_dict(t) -> dict:
    if isinstance(t, dict):
        return t
    try:
        return t.model_dump()
    except Exception:  # noqa: BLE001
        return {}


def _pair_sim(a: dict, b: dict) -> float:
    """Сходство двух кандидатов по метаданным (0..1): тот же артист > жанр."""
    aa = (a.get("artist") or "").strip().lower()
    ab = (b.get("artist") or "").strip().lower()
    if aa and aa == ab:
        return 1.0
    ga = set(a.get("genres") or [])
    gb = set(b.get("genres") or [])
    if ga and gb and (ga & gb):
        return 0.5
    # общий токен в артисте («Баста» vs «Баста, Моя Мишель» — коллабы)
    ta = {w for w in aa.replace(",", " ").split() if len(w) >= 4}
    tb = {w for w in ab.replace(",", " ").split() if len(w) >= 4}
    if ta and tb and (ta & tb):
        return 0.7
    return 0.0


def mmr_rerank(
    tracks: list,
    scores: dict[str, float] | None = None,
    *,
    seed_artist: str = "",
    lambda_: float = 0.72,
    per_artist_seed: int = 5,
    per_artist_other: int = 3,
) -> list:
    """MMR-перестановка (ТЗ similar п.6). Возвращает ВСЕ треки (выбранные
    по MMR первыми, не поместившиеся — в хвост, кандидаты не теряются).

    scores — {track_id: rel 0..1} (из rank() или CLAP); трек без score
    получает 0.5 («не знаем» — не занижаем, ТЗ п.5).
    """
    if not tracks or len(tracks) <= 2:
        return tracks
    seed_n = (seed_artist or "").strip().lower()
    dlist = [_as_dict(t) for t in tracks]
    rels: list[float] = []
    for d in dlist:
        s = (scores or {}).get(d.get("id") or "", None)
        rels.append(0.5 if s is None else max(0.0, min(1.0, float(s))))

    picked_idx: list[int] = []
    artist_count: dict[str, int] = {}
    used: set[int] = set()
    n = len(dlist)
    while len(used) < n:
        best_i, best_v = -1, -1e9
        for i in range(n):
            if i in used:
                continue
            d = dlist[i]
            a = (d.get("artist") or "").strip().lower()
            is_seed = bool(seed_n and a and (seed_n in a or a in seed_n))
            cap = per_artist_seed if is_seed else per_artist_other
            if artist_count.get(a, 0) >= cap:
                continue
            redundancy = 0.0
            if picked_idx:
                redundancy = max(_pair_sim(d, dlist[j]) for j in picked_idx)
            v = lambda_ * rels[i] - (1.0 - lambda_) * redundancy
            # стабильность: при равенстве — исходный порядок (rel rank()-а)
            if v > best_v + 1e-9:
                best_v, best_i = v, i
        if best_i < 0:
            break  # все под лимитами артистов
        used.add(best_i)
        picked_idx.append(best_i)
        a = (dlist[best_i].get("artist") or "").strip().lower()
        artist_count[a] = artist_count.get(a, 0) + 1
    tail = [i for i in range(n) if i not in used]
    # Хвост (не прошедший лимиты артиста): ЧУЖИЕ первыми, seed-артист — в самый
    # конец. Иначе 6-я «Баста» из хвоста просочится в топ-limit (ТЗ п.6:
    # «не более 3-5 исходного исполнителя» должно действовать и на выдачу).
    seed_tail = [i for i in tail if seed_n and seed_n in
                 ((dlist[i].get("artist") or "").strip().lower())]
    other_tail = [i for i in tail if i not in set(seed_tail)]
    out = [tracks[i] for i in picked_idx] + [tracks[i] for i in other_tail] \
        + [tracks[i] for i in seed_tail]
    logger.info(
        "mmr_rerank: %d tracks, выбрано MMR=%d, seed_artist=%r, лимиты %d/%d",
        n, len(picked_idx), seed_n[:30], per_artist_seed, per_artist_other,
    )
    return out
