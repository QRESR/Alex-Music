"""Yandex Music client — свежий каталог (source of new tracks).

Почему так (2026-10-29):
- Яндекс API блокирует IP датацентра кодом **451 Unavailable For Legal
  Reasons** (поиск/метаданные). Через residential HTTP-прокси (российский
  IPv4) — 200. Токен валидный (device-flow из MA).
- **Файлы треков (storage.yandex.net) и обложки (avatars.yandex.net)
  доступны НАПРЯМУЮ с сервера** (проверено: 206 audio/mpeg 192kbps) —
  прокси НЕ нужен для стриминга → 10Mbps прокси не узкое место,
  качаем на полной скорости сервера.
- `aiohttp.request()` внутри библиотеки yandex-music имеет
  trust_env=False → env-прокси в MA не сработал бы; поэтому ищем
  в СВОЁМ бэкенде через httpx(proxy=...).

Формат direct-link (из yandex_music/download_info.py):
    sign = md5('XGRlBW9FXlekgbPrRHuSiA' + path[1:] + s)
    url  = https://{host}/get-mp3/{sign}/{ts}{path}
Ссылка живёт ~60с (иначе 410) → получать ПЕРЕД каждым скачиванием.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import asyncio
from pathlib import Path

import httpx

logger = logging.getLogger("ym_client")

_BASE_DIR = Path(__file__).resolve().parent.parent
_PRIVATE = _BASE_DIR / "private"

API_BASE = "https://api.music.yandex.net"
_SIGN_SALT = "XGRlBW9FXlekgbPrRHuSiA"
_YM_ID_PREFIX = "ym_"

# ---------------------------------------------------------------- secrets --
def _read_secret(name: str) -> str:
    try:
        return (_PRIVATE / name).read_text().strip()
    except OSError:
        return ""


TOKEN = _read_secret("yandex-token.txt")
PROXY = _read_secret("yandex-proxy.txt")  # http://user:pass@host:port

# ------------------------------------------------------------ http client --
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Клиент ДЛЯ МЕТАДАННЫХ: через российский прокси (иначе 451)."""
    global _client
    if _client is None or _client.is_closed:
        headers = {"Authorization": f"OAuth {TOKEN}"} if TOKEN else {}
        _client = httpx.AsyncClient(
            proxy=PROXY or None,
            headers=headers,
            timeout=httpx.Timeout(20.0, connect=15.0),
            follow_redirects=True,
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def is_ym_id(track_id: str) -> bool:
    return track_id.startswith(_YM_ID_PREFIX)


# ------------------------------------------------------------------ search --
def _cover_url(cover_uri: str | None) -> str:
    """avatars.yandex.net/.../%% → https://.../400x400."""
    if not cover_uri:
        return ""
    if cover_uri.startswith("http"):
        return cover_uri.replace("%%", "400x400")
    return f"https://{cover_uri.replace('%%', '400x400')}"


# ------------------------------------------------- транслит для поиска ------
# Поиск «кривой»: Яндекс лучше ранжирует по ТОЙ раскладке, в которой каталог
# записан. Русские артисты в Яндексе часто имеют и латинские алиасы, и наоборот:
# «Баста» находит лучше как «basta», «кис-кис» — как «kis-kis». Поэтому ищем
# в ОБЕ стороны и объединяем, а при пустом — по слову/подслову.
#
# Пары даны «по-русски» (как звучит/пишется в каталоге) — и это то, что нужно
# для кириллица→латиница; обратное — инверсия той же таблицы.

# многосимвольные (сначала, чтобы «дж» не разложилось на «д»+«ж»)
_PAIRS_MULTI = [
    ("дж", "dzh"), ("шч", "shch"), ("дж", "dj"),
    ("ье", "ye"), ("ья", "ya"), ("ю", "yu"), ("я", "ya"),
    ("ч", "ch"), ("ш", "sh"), ("щ", "shch"), ("ж", "zh"),
    ("х", "kh"), ("ц", "ts"), ("ё", "yo"),
]
# одиночные
_PAIRS_SINGLE = [
    ("а", "a"), ("б", "b"), ("в", "v"), ("г", "g"), ("д", "d"), ("е", "e"),
    ("з", "z"), ("и", "i"), ("й", "y"), ("к", "k"), ("л", "l"), ("м", "m"),
    ("н", "n"), ("о", "o"), ("п", "p"), ("р", "r"), ("с", "s"), ("т", "t"),
    ("у", "u"), ("ф", "f"), ("ы", "y"), ("э", "e"),
]


def _build_maps() -> tuple[dict[str, str], dict[str, str]]:
    """(cyr→lat, lat→cyr) на основе ОДНОЙ парной таблицы."""
    cyr2lat: dict[str, str] = {}
    lat2cyr: dict[str, str] = {}

    def add(cyr: str, lat: str) -> None:
        cyr2lat[cyr] = lat
        # обратная (регистронезависимо; латиница всегда нижний, верх через isupper)
        if lat not in lat2cyr:
            lat2cyr[lat] = cyr

    # многосимвольные ПЕРВЫМИ (и в lat2cyr — тоже, чтобы «shch» не разбирался)
    for cyr, lat in _PAIRS_MULTI:
        add(cyr, lat)
    for cyr, lat in _PAIRS_SINGLE:
        add(cyr, lat)
    # ещё «ь/ъ» в латиницу (мягкий/твёрдый знак — пустые, но чтобы не ломали)
    cyr2lat["ь"] = ""
    cyr2lat["ъ"] = ""
    return cyr2lat, lat2cyr


_CYR2LAT, _LAT2CYR = _build_maps()

# латиница→кириллица: «shch» → «щ», «ya» → «я», «d» → «д» ... с жадным
# сопоставлением по длине (многосимвольные раньше).
_LAT_SORTED = sorted(_LAT2CYR.items(), key=lambda kv: -len(kv[0]))

# Одиночные латинские буквы, которые МОГТ быть русскими (для фильтра «звучит
# по-русски»): если в запросе есть латинская буква ВНЕ этой множества (h, w, x,
# q, z для англ. слов) — не транслитим (иначе «night» → мусор «нигhт»).
_LAT2CYR_SINGLE = {lat for lat in _LAT2CYR if len(lat) == 1}


def translit_to_latin(text: str) -> str:
    """кириллица → латиница («Баста» → «Basta», «кис-кис» → \"kis-kis\")."""
    out: list[str] = []
    i = 0
    s = text
    n = len(s)
    while i < n:
        ch = s[i]
        low = ch.lower()
        matched = False
        # попробовать 2-символьный («дж») потом 1
        for width in (2, 1):
            chunk = s[i:i + width].lower()
            if len(chunk) == width and chunk in _CYR2LAT:
                lat = _CYR2LAT[chunk]
                if ch.isupper() and lat:
                    lat = lat[0].upper() + lat[1:]
                out.append(lat)
                i += width
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1
    return "".join(out)


def translit_to_cyrillic(text: str) -> str:
    """латиница → кириллица («Basta» → «Баста», «kis-kis» → «кис-кис»)."""
    out: list[str] = []
    i = 0
    s = text
    n = len(s)
    while i < n:
        ch = s[i]
        low = ch.lower()
        matched = False
        # жадно: самый длинный латинский токен, который есть в таблице
        for lat, cyr in _LAT_SORTED:
            if s[i:i + len(lat)].lower() == lat:
                if ch.isupper():
                    cyr = cyr[0].upper() + cyr[1:]
                out.append(cyr)
                i += len(lat)
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1
    return "".join(out)


def _has_cyr(s: str) -> bool:
    return bool(re.search(r"[а-яё]", s, re.I))


def _has_lat(s: str) -> bool:
    return bool(re.search(r"[a-z]", s, re.I))


def _query_variants(query: str) -> list[str]:
    """Варианты запроса: оригинал + транслит в ту сторону, в которой НЕ написан.

    «Баста» → [«Баста», «basta»]; «basta» → [«basta», «Баста»].
    Смешанный или нерусское слово («night») → только оригинал (транслитит
    мусор: «night» → «нигhт» — хуже, чем не искать).
    """
    q = (query or "").strip()
    if not q:
        return []
    variants = [q]
    cyr, lat = _has_cyr(q), _has_lat(q)
    if cyr and not lat:
        tr = translit_to_latin(q)
        if tr and tr.lower() != q.lower():
            variants.append(tr)
    elif lat and not cyr:
        # Транслитим латиницу→кириллицу ТОЛЬКО если это «звучит по-русски»,
        # т.е. ВСЕ латинские буквы распознаются как русские (basta, kino,
        # asti). «night» содержит h (не русская) → транслит дал бы мусор
        # «нигhт» → не транслитим, ищем как есть.
        tr = translit_to_cyrillic(q)
        has_unknown = any(
            ch.isalpha() and ch.isascii() and ch.lower() not in _LAT2CYR_SINGLE
            for ch in q
        )
        if tr and _has_cyr(tr) and not has_unknown:
            variants.append(tr)
    return variants


def _track_features(t: dict) -> dict:
    """Извлечение характеристик трека из ответа Яндекса (ТЗ п.3.3, шаг 4).

    Только то, что реально есть в API; отсутствующее → пусто/0
    (ТЗ п.4: не подменять неизвестные выдуманными значениями).
    """
    year = 0
    genres: list[str] = []
    album = ""
    explicit = bool(t.get("explicit") or False)
    popularity = 0
    loudness = 0.0
    try:
        albums = t.get("albums") or []
        if albums:
            a0 = albums[0]
            year = int(a0.get("year") or 0)
            album = a0.get("title") or ""
            g = a0.get("genre")
            if g:
                genres.append(str(g))
            lc = a0.get("likesCount")
            if isinstance(lc, (int, float)):
                popularity = int(lc)
    except (ValueError, TypeError, IndexError):
        pass
    # жанры артиста (у некоторых исполнителей albums.genre пуст, но artists.genres есть)
    for a in (t.get("artists") or [])[:2]:
        for g in (a.get("genres") or [])[:4]:
            gs = str(g)
            if gs and gs not in genres:
                genres.append(gs)
    # r128: {"i": громкость dB, "tp": true-peak} — реальная характеристика звучания
    r = t.get("r128")
    if isinstance(r, dict):
        try:
            loudness = float(r.get("i") or 0.0)
        except (TypeError, ValueError):
            loudness = 0.0
    return {
        "year": year,
        "genres": genres[:6],
        "album": album,
        "explicit": explicit,
        "popularity": popularity,
        "loudness": loudness,
    }


async def _ym_search_once(query: str, limit: int) -> list[dict]:
    """Один запрос к Яндексу (без транслита). list[dict] или []."""
    resp = await _get_client().get(
        f"{API_BASE}/search",
        params={"type": "track", "text": query, "page": 0},
    )
    if resp.status_code != 200:
        logger.warning("Yandex search HTTP %d for %r", resp.status_code, query)
        return []
    data = resp.json().get("result") or {}
    items = ((data.get("tracks") or {}).get("results")) or []
    out: list[dict] = []
    for idx, t in enumerate(items[:limit]):
        if not t.get("available"):
            continue
        artists = ", ".join(a.get("name", "") for a in (t.get("artists") or []))
        tid = t.get("id")
        _feat = _track_features(t)
        out.append(
            {
                "id": f"{_YM_ID_PREFIX}{tid}",
                "title": t.get("title") or "",
                "artist": artists,
                "duration": int((t.get("durationMs") or 0) // 1000),
                "thumbnail": _cover_url(t.get("coverUri")),
                "channel": "",
                "url": f"https://music.yandex.ru/track/{tid}",
                "year": _feat["year"],
                # позиция в выдаче Яндекса — PROXY популярности: движок Яндекса
                # сам взвешивает популярность+релевантность, чем выше — тем лучше.
                "rank": idx,
                # Патч 2 шаг 4: расширенные характеристики (если есть в API)
                "genres": _feat["genres"],
                "album": _feat["album"],
                "explicit": _feat["explicit"],
                "popularity": _feat["popularity"],
                "loudness": _feat["loudness"],
            }
        )
    return out


async def search(query: str, limit: int = 20) -> list[dict]:
    """Поиск треков в Яндекс Музыке (с транслитом и пословным фолбэком).

    Стратегия (как у «Алисы» — дать релевантный результат, а не пустоту):
      1) оригинал + транслит в обратную сторону («Баста»→«basta»);
      2) при пустом — ПО СЛОВУ (многословный запрос часто портит ранжирование);
      3) РАНЖИРОВАНИЕ ПО ИСПОЛНИТЕЛЮ: «Асти» → ARTIK & ASTI в топе, а не
         одноимённые треки чужих (Брутто, ALIB, Cali); точное совпадение
         имени артиста поднимается вверх;
      4) объединяем (dedup по id), не теряя порядок.
    """
    if not TOKEN:
        logger.info("Yandex: no token, skip search")
        return []
    try:
        attempts = _query_variants(query)
        if not attempts:
            attempts = [query]
        # пословный фолбэк (только если первый путь дал мало)
        words = [w for w in re.split(r"\s+", (query or "").strip()) if len(w) >= 3]
        if len(words) > 1:
            for w in words:
                for v in _query_variants(w):
                    if v not in attempts:
                        attempts.append(v)

        seen: set[str] = set()
        raw: list[dict] = []
        # ФИКС «рэп-подборка даёт джаз/Задорнова»: при ЖАНОВОМ запросе слово-якорь
        # «рэп» отрезалось лимитом attempts[:4] (каждое слово раздувается транслитом:
        # ['зарубежный рэп 2025 хиты', транслит, 'зарубежный', 'zarubezhnyy'] —
        # «рэп» не попадал вовсе → в пуле НИ ОДНОГО рэп-трека, один шлак).
        # Теперь якоря ПЕРЕД словами: [запрос, «<качеств.> <жанр>», жанр, EN-жанр].
        genre = _query_genre(query)
        if genre:
            q_words = [w for w in re.split(r"\s+", (query or "").strip()) if len(w) >= 3]
            quals = [w for w in q_words if w.startswith(("зарубеж", "русск", "отечеств", "зарубеж"))]
            anchor = " ".join(dict.fromkeys(quals + [genre]))       # «зарубежный рэп»
            en = _GENRE_EN.get(genre)
            front = [query, anchor, genre] + ([en] if en and en != genre else [])
            front = [f for f in dict.fromkeys(front) if f]
            attempts = front + [a for a in attempts if a not in front]
        # Опрашиваем ВСЕ варианты (оригинал + транслит), не прерываясь на
        # первом: «Баста» (кириллица) даёт мусор, а транслит «Basta» — топ
        # (Баста - Выпускной). Прежний ранний выход отсекал транслит → провал.
        # ПАРАЛЛЕЛЬНЫЙ ПОИСК (п.15 ТЗ): до 4 попыток (оригинал+транслит+
        # пословный фолбэк) шлём РАЗОМ через gather, а не последовательно
        # (было: 4 × ~0.7-1.5с ≈ 3-5с; стало ≈ max одной попытки).
        # Результат собираем В ПОРЯДКЕ attempts (оригинал первым), чтобы
        # _rank_by_artist(ранжирование по артисту) не поехали.
        _plans = attempts[:4] if len(attempts) > 1 else attempts  # потолок, чтобы не долбить прокси
        if not _plans:
            _results = []
        else:
            from . import concurrency as _cc_ym
            async def _attempt(a: str) -> list[dict]:
                async with _cc_ym.source_gate():
                    return await _ym_search_once(a, limit)
            # единый вид: list[list[dict]] (gather отдаёт по одному list на попытку)
            _results = await asyncio.gather(
                *(_attempt(a) for a in _plans), return_exceptions=True)
        # НЕ прерываемся на первом варианте: транслит (второй) часто даёт
        # лучший топ («Баста»→мусор, «Basta»→Баста). Все попытки → общий пул.
        for _r in _results:
            if isinstance(_r, Exception):
                logger.warning("Yandex attempt failed: %s", _r)
                continue
            for h in _r:
                if h["id"] not in seen:
                    seen.add(h["id"])
                    raw.append(h)

        # ЛИМИТ строго: «десяти хватит из поиска», подборка — 10.
        # Прежнее [:limit*2] отдавало 15-20 при limit=10 (жалоба на мусор в хвосте).
        out = _rank_by_artist(raw, query)[: limit]
        logger.info(
            "Yandex search %r -> %d hits (attempts=%s)", query, len(out), attempts[:3]
        )
        return out
    except Exception as e:  # noqa: BLE001 — источник не должен ронять /search
        logger.warning("Yandex search failed for %r: %s", query, e)
        return []


def _norm_artist(s: str) -> str:
    """Нормализация + ДВУСТОРОННИЙ транслит для сравнения.

    Ключевое: запрос «Асти» (кириллица) должен матчиться с «ARTIK & ASTI»
    (латиница) — иначе совпадают только одноимённые мусорные треки. Поэтому
    и имя, и запрос прогоняются через транслит в обе стороны, регистр глушим.
    """
    low = (s or "").lower()
    # транслит в латиницу и в кириллицу → сравниваем «универсальную» форму
    try:
        from .ym_client import translit_to_latin
        low_l = translit_to_latin(low)
    except Exception:  # noqa: BLE001
        low_l = low
    return re.sub(r"[^0-9a-zа-яё]+", " ", low_l).strip()


def _tokens(s: str) -> set[str]:
    """Слова нормализованной строки (для точного токен-совпадения)."""
    return {t for t in (s or "").split() if len(t) >= 3}


# ЖАНРЫ (для реранга «жанровой подборки»): токен запроса ∈ словарь → это жанр.
_GENRE_TOKENS = {
    # рэп/хип-хоп
    "рэп", "рэпом", "рэпа", "рэповый", "рэповая", "рэповые",
    "реп", "репом", "репа", "рэпчик",
    "хипхоп", "хип", "хоп", "hiphop", "hip", "hop",
    # рок/поп
    "поп", "попс", "попса", "попса", "рок", "рокнролл", "рок-н-ролл",
    "альтернатив", "альтернатива", "инди", "панк", "эмок", "металкор",
    # танцевальная/электронная (Алиса: «танцевальную», «электронную»)
    "электрон", "электронная", "электронную", "электронного", "electronic",
    "танцев", "танцевальная", "танцевальную", "танцевального", "dance",
    "хаус", "house", "техно", "techno", "транс", "trance", "рейв", "rave",
    "дром", "drumandbass", "dubstep", "EDM", "едм", "даунтемп",
    "диско", "disco", "фонк", "фьючерсбас", "эмбиент", "ambient",
    # релакс/спокойствие (Алиса: «релакс», «спокойная»)
    "релакс", "релаксация", "расслаб", "спокойн", "мягк", "тихий", "тихая",
    "лойфай", "лоуфай", "лоу-фай", "лоуфай", "lofi", "lo-fi", "чилл", "chill",
    # романтическая (Алиса: «романтическую»)
    "романтик", "романтическая", "романтическую", "романтического", "нежн",
    "любовн", "мэллоу", "lovesong",
    # жанры (общие)
    "попса", "джаз", "jazz", "блюз", "blues", "соул", "soul", "фанк", "funk",
    "кантри", "country", "шансон", "фолк", "folk", "регги", "reggae",
    "латино", "latin", "классическ", "классика", "opera", "опера",
    "рэп", "кантри",
    # энергия/бодрость (частые «для дороги/зала»)
    "бодр", "энергичн", "динамичн", "боев", "эпичн",
    # настроения (часто как подборка)
    "весел", "весёл", "грустн", "мрачн", "радостн", "меланхол",
    # ЛАТИНСКИЕ жанры (originQuery может быть латиницей: rap/rock/pop из
    # _GENRE_EN — без них _query_genre('rap')=None → жанр-якорь брался из ctx).
    "rap", "rock", "pop", "metal", "indie", "punk", "phonk",
    "alternative", "relax", "romantic", "chanson", "classical", "calm",
    "hip-hop",
}
# Слова-мусор, которые не жанр (но могут содержать жанр-подстроку).
_GENRE_NOISE = {"трек", "треки", "песня", "песни", "музыка", "подборка", "микс", "включи"}

# БЛОК-ЛИСТ «старья/мусора» (жалоба: «Macarena, Karaoke in the Sky — никогда»).
# Совпадение по title+artist (lower, без ё). Глобально: search/radio/similar/rotor.
# Макарена (Los Del Rio) — вечная попса, Караоке — кавер-мусор. ДОПОЛНЯТЬ.
_BLOCKED = {
    "macarena",          # Los Del Rio — «Macarena» (никогда не включать)
    "макарена",          # кириллица: «Сергей Макарена» и т.п.
    "karaoke in the sky",
    "караоке",
    "karaoke",
}

def _is_blocked(t: dict) -> bool:
    """True, если трек попал в блок-лист мусора (title или artist)."""
    blob = f"{t.get('title', '')} {t.get('artist', '')}".lower().replace("ё", "е")
    return any(b in blob for b in _BLOCKED)

# RU-основа → EN-форма (для _genre_score: запрос «electronic» → русский трек
# «Электронная музыка» и жанр Яндекса «electronic» — оба должны совпасть).
# Транслит НЕ работает («электронная»→elektronnaya ≠ electronic) — нужен словарь.
_GENRE_EN = {
    "электрон": "electronic", "танцевальн": "dance", "танцев": "dance",
    "романтическ": "romantic", "релакс": "relax", "спокойн": "calm",
    "рэп": "rap", "реп": "rap", "хипхоп": "hip-hop",
    "рок": "rock", "поп": "pop", "попса": "pop", "джаз": "jazz",
    "блюз": "blues", "соул": "soul", "фанк": "funk", "кантри": "country",
    "шансон": "chanson", "фолк": "folk", "регги": "reggae",
    "латино": "latin", "классическ": "classical", "опера": "opera",
    "метал": "metal", "инди": "indie", "альтернатив": "alternative",
    "техно": "techno", "хаус": "house", "транс": "trance", "диско": "disco",
    "эмбиент": "ambient", "фонк": "phonk", "панк": "punk", "рейв": "rave",
    "лойфай": "lo-fi", "лоуфай": "lo-fi", "лоу-фай": "lo-fi",
    "чилл": "chill", "лэтм": "latin",
}


def _query_genre(query: str) -> str | None:
    """Вернуть жанр-токен из запроса, если это ЖАНОВОЙ запрос («рэп», «хип-хоп»).

    ТОЧНОЕ совпадение токена со словарём (после нормализации «хип-хоп»→«хипхоп»),
    чтобы «рэпорт»/«репутация» НЕ считались жанром «рэп» (ловил startswith-баг).
    """
    q = (query or "").lower().replace("-", "").replace("ё", "е")
    q = re.sub(r"[^0-9a-zа-я]+", " ", q)
    # Типичные русские суффиксы склонения/прилагательного — отрезаем до основы,
    # чтобы «электронную»→«электрон», «танцевальную»→«танцевальн» (иначе
    # _genre_score ищет полную форму «электронную» в треке и не находит).
    _SUF = ("ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий",
            "ой", "ая", "ему", "ому", "ими", "ыми", "ов", "ев", "ам", "ям",
            "ах", "ях", "ом", "ем", "ой", "ей", "ой", "у", "ю", "а", "я", "ы", "и", "е")
    matches: list[str] = []
    for w in q.split():
        if w in _GENRE_NOISE:
            continue
        # сначала — точное совпадение словаря (латиница вроде dance/edm и основы)
        if w in _GENRE_TOKENS:
            # отрезаем суффикс от склонённой формы («электронную»→«электрон»)
            got = None
            for s in sorted(_SUF, key=len, reverse=True):
                if len(w) > len(s) + 3 and w.endswith(s):
                    stem = w[: -len(s)]
                    if stem in _GENRE_TOKENS or len(stem) >= 5:
                        got = stem
                        break
            if got is None:
                got = w   # уже основа / латиница («edm», «jazz»)
            if got not in matches:
                matches.append(got)
            continue
        # стем-совпадение: «электронную» → основа «электрон» есть в словаре
        got = None
        for s in sorted(_SUF, key=len, reverse=True):
            if len(w) > len(s) + 3 and w.endswith(s):
                stem = w[: -len(s)]
                if stem in _GENRE_TOKENS:
                    got = stem
                    break
        if got:
            if got not in matches:
                matches.append(got)
            continue
        # множественные формы рэп/рок: «рэпом»/«рэпа» → основа, НО не «рэпорт»
        for g in ("рэп", "реп", "рок", "поп", "техно", "хаус", "метал", "джаз"):
            if w.startswith(g) and len(w) > len(g) and w[len(g):] in ("ом", "а", "у", "ы", "и", "е", "ой", "ая", "ое", "ые", "ие", "ов", "ев", "ам", "ям", "ах", "ях", "ей"):
                if g not in matches:
                    matches.append(g)
                break
    if not matches:
        return None
    # ПРИОРИТЕТ ЯКОРЯ (ТЗ п.9): «классический/классика» — модификатор ВОЗРАСТА,
    # а не самостоятельный жанр, когда рядом конкретный жанр. Иначе
    # «классический русский рок 90-х» якорился как classical → оркестры
    # вместо рока (проверено 2026-10-31).
    _modifiers = {"классическ", "классик", "классика", "старинн", "ретро"}
    specific = [m for m in matches if m not in _modifiers]
    return (specific or matches)[0]


def _genre_score(t: dict, genre: str) -> int:
    """0 — трек про ЖАНР (совпал жанр в названии/артисте/жанрах Яндекса);
    1 — не подтверждён жанр (иначе «Рэп» как часть названия чужого трека
    топит мусор). Ищем и RU-основу, и EN-форму (запрос «electronic» →
    русский трек «Электронная музыка» / жанр Яндекса «electronic»)."""
    g = genre.lower().replace("-", "")
    # кандидаты: RU-основа + EN-форма (обе проверяем подстрокой в тексте)
    cands = {g}
    en = _GENRE_EN.get(g)
    if en:
        cands.add(en.replace("-", ""))
    blob = f"{t.get('title','')} {t.get('artist','')}".lower()
    ym_genres = " ".join(t.get("genres") or []).lower() if isinstance(t.get("genres"), list) else str(t.get("genres") or "")
    hay = f"{blob} {ym_genres}".replace("-", "")
    for c in cands:
        if c and len(c) >= 3 and (c in hay or c in ym_genres):
            return 0
    return 1


def pop_score(t: dict) -> int:
    """Гибрид СВЕЖЕСТЬ + ПОПУЛЯРНОСТЬ (жалоба: «не только по дате, но и
    по популярности»). Год решает сильно (×5), позиция в выдаче Яндекса
    (rank — proxy популярности, движок сам взвешивает популярность) даёт
    до ~20 очков → очень популярный трек может обойти на пару лет старше.
    Неизвестный год (0) считаем ~2019, чтобы не топил в конец.
    Module-level: используется и в _rank_by_artist (поиск), и в /radio
    (фильтр качества рекомендаций). Rotor-треки БЕЗ rank → год работает.
    """
    y = t.get("year") or 2019
    return (2026 - y) * 5 + int(t.get("rank", 20))


# ----------------------------------------------------- якорь (жанр+артист) ----
def _artist_anchor_from_ctx(ctx: str) -> str:
    """Артист-якорь из ctx «Артист - Название» (или note «баста»).

    ЖАЛОБА 2026-10-30: ротор от «Баста — Выпускной» давал Danza Kuduro(2015)/
    Авария(2013)/Crazy Frog(2005)/Queen(2000) — потому что _query_genre('баста')
    =None (это артист, не жанр) → якоря не было → ротор шёл вслепую.
    Теперь имя артиста ИЗ ctx/seed — якорь: похожие держатся этого артиста.
    Возвращает нижний регистр имени («баста») или "" если не разобрано.
    """
    if not ctx:
        return ""
    s = ctx.strip()
    # формат «Артист - Название» (первый дефис-разделитель)
    if " - " in s:
        artist = s.split(" - ", 1)[0]
    else:
        artist = s   # одиночное имя (note «баста»)
    artist = artist.strip().lower().replace("ё", "е")
    # чистим слова-мусор, оставляя только токены-имена
    artist = re.sub(
        r"\b(трек|треки|песн\w*|музык\w*|включи|включить|поставь|поиграй|"
        r"найди|подборк\w*|похожее|похожая|похожие|на|это|этот|эта)\b",
        " ", artist,
    )
    artist = re.sub(r"\s+", " ", artist).strip()
    return artist if artist else ""


def match_artist_anchor(t: dict, anchor: str) -> bool:
    """True, если трек относится к артисту-якорю (релевантен подборке).

    Сравниваем по токенам (нормализация + транслит): «баста»↔«Basta»,
    точное вхождение имени в список артистов (учитываем «ARTIK & ASTI»
    содержит «asti»)."""
    if not anchor:
        return False
    a_tok = _tokens(_norm_artist(anchor))
    if not a_tok:
        return False
    track_art = _tokens(_norm_artist(t.get("artist", "")))
    if a_tok & track_art:
        return True
    # имя-подстрока («асти» в «ANNA ASTI» / «ARTIK & ASTI»)
    return _norm_artist(anchor) in _norm_artist(t.get("artist", ""))


def is_old(t: dict) -> bool:
    """True, если трек заведомо СТАРЫЙ (режется из рекомендаций).

    ПОРОГ 5 ЛЕТ (Алекс 2026-10-30): «старшие 5 лет в топку, если сам не
    выбрал». Треки старше 5 лет НЕ рекомендуем в «похожих/подборках» —
    но точный поиск (юзер сам попросил «Queen - We Will Rock You») НЕ
    режется. Треки БЕЗ года (0) НЕ режем — могут быть свежие."""
    y = t.get("year") or 0
    if y <= 0:
        return False
    _cutoff = time.gmtime(time.time()).tm_year - 5
    return y < _cutoff



def _rank_by_artist(tracks: list[dict], query: str) -> list[dict]:
    """Ранжирование выдачи: релевантность артисту → затем СВЕЖЕСТЬ (год).
    Проблема «шлака в топе» (жалоба): совпадение СЛОВА в НАЗВАНИИ трека
    раздувает мусор — «basta» → чужие «Basta»/Celentano, «kino» → топ из
    случайных «Кино». Лечим:
      1) релевантность:
         score 0 — имя артиста содержит токен запроса → РЕАЛЬНЫЙ исполнитель;
         score 1 — нет совпадения артиста → НЕ поднимать по названию (в конец,
                   отсортированные по свежести — «kino» не топит случайные);
      2) внутри уровня — СВЕЖЕСТЬ: новее год → выше (неизвестный год — низ).
    Слова-подсказки вида «трек/песня» отфильтровываются ДО ранжирования,
    иначе lfm-запрос «кино трек» → токен «трек» в чужих названиях.
    """
    if not tracks:
        return tracks
    q_norm = _norm_artist(query)
    # убрать слова-мусор («трек», «песня», «включи») — иначе они и дают «шлак»
    q_norm = re.sub(
        r"\b(трек|треки|песн\w*|музык\w*|включи|включить|поставь|поиграй|найди|подборк\w*)\b",
        " ",
        q_norm,
    )
    q_tok = _tokens(q_norm)

    def rel(t: dict) -> int:
        if not q_tok:
            return 0
        artist_tok = _tokens(_norm_artist(t.get("artist", "")))
        if q_tok & artist_tok:
            return 0
        return 1   # без совпадения артиста — в конец (не по названию!)

    def freshness(t: dict) -> int:
        y = t.get("year") or 0
        return -y if y > 0 else 0x7FFFFFFF   # неизвестный год — в конец группы

    ordered = sorted(tracks, key=lambda t: (rel(t), pop_score(t), freshness(t)))
    # ФИКС «жанр → мусор/шлак»: если это ЖАНОВОЙ запрос («рэп»), поднимаем
    # треки, подтверждающие жанр (название/артист/жанры Яндекса), вверх —
    # «Качалка Рэп»/«для спортзала» уходят ниже настоящего рэпа.
    g = _query_genre(query)
    if g:
        # ФИКС «зарубежный рэп → русский рэп в топе»: при запросе с качеством
        # «зарубежн…» штрафуем кириллических артистов (и наоборот для «русск…») —
        # иначе Тимати/Скриптонит (score=0) обгоняют Eminem по свежести.
        ql = (query or "").lower().replace("ё", "е")
        want_foreign = "зарубеж" in ql or "foreign" in ql
        want_russian = "русск" in ql or "отечеств" in ql

        def geopen(t: dict) -> int:
            a = (t.get("artist") or "").lower()
            has_cyr = any(("а" <= ch <= "я") or ch == "ё" for ch in a)
            has_lat = any(ch.isascii() and ch.isalpha() for ch in a)
            if want_foreign and has_cyr and not has_lat:
                return 1        # «зарубежный» запрос, а артист русскоязычный
            if want_russian and has_lat and not has_cyr:
                return 1        # «русский» запрос, а артист латиницей
            return 0

        ordered = sorted(ordered, key=lambda t: (_genre_score(t, g), geopen(t), rel(t), pop_score(t), freshness(t)))
    # Блок-лист мусора (Macarena и т.п.) — выкидываем ДО возврата (search/radio).
    _before = len(ordered)
    ordered = [t for t in ordered if not _is_blocked(t)]
    if len(ordered) != _before:
        logger.info("blocklist: %d -> %d (убран мусор)", _before, len(ordered))
    # БЕНЧ 2026-10-31 (100 запросов): в топ-10 лезли ЗАПИСИ РАДИОСТАНЦИЙ
    # с пустым артистом — «DFM Дискач 90-х», «93,6 JAM FM Russkiye Khity»,
    # «European Hit Radio», «Record 80-х», «Autoradio», «fuego stereo» —
    # на запросы «песни 80-х / хиты 2000-х / испанские хиты / музыка на
    # 8 марта» они были top-1. Настоящий трек у Яндекса ВСЕГДА имеет
    # артиста (даже «Various Artists»), пустой artist = не-трек → режем.
    _no_empty = [t for t in ordered if (t.get("artist") or "").strip()]
    if len(_no_empty) != len(ordered):
        logger.info("radio-junk: %d -> %d (записи без артиста убраны)",
                    len(ordered), len(_no_empty))
        ordered = _no_empty
    return ordered


# ------------------------------------------------------- похожие треки ----
def _track_year(t: dict) -> int:
    """Год релиза из albums[0].year (0 = неизвестен). Для ранжирования по свежести."""
    try:
        albums = t.get("albums") or []
        return int(albums[0].get("year") or 0) if albums else 0
    except (ValueError, TypeError, IndexError):
        return 0


# Движок рекомендаций Яндекс Музыки (Rotor): GET /tracks/{id}/similar →
# result.similarTracks[] — ПОХОЖИЕ треки, независимые от результатов поиска.
# Это то, из чего берётся «следующий трек» (как «Моя волна»/радио), а не
# из выдачи поиска. Через прокси (метаданные) — проверено: 200, 18 items.
async def similar_tracks(track_id: str, limit: int = 30) -> list[dict]:
    """Похожие треки для ym_<id> (рекомендательный пул «радио»)."""
    if not track_id.startswith(_YM_ID_PREFIX):
        return []
    raw_id = track_id[len(_YM_ID_PREFIX):]
    try:
        resp = await _get_client().get(
            f"{API_BASE}/tracks/{raw_id}/similar",
            params={"limit": max(limit, 10)},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ym similar failed for %s: %s", track_id, e)
        return []
    if resp.status_code != 200:
        logger.warning("ym similar HTTP %d for %s", resp.status_code, track_id)
        return []
    res = resp.json().get("result") or {}
    items = res.get("similarTracks") or []
    out: list[dict] = []
    for t in items[:limit]:
        if not t.get("available"):
            continue
        tid = t.get("id")
        if tid is None:
            continue
        _feat = _track_features(t)
        out.append(
            {
                "id": f"{_YM_ID_PREFIX}{tid}",
                "title": t.get("title") or "",
                "artist": ", ".join(a.get("name", "") for a in (t.get("artists") or [])),
                "duration": int((t.get("durationMs") or 0) // 1000),
                "thumbnail": _cover_url(t.get("coverUri")),
                "channel": "",
                "url": f"https://music.yandex.ru/track/{tid}",
                "year": _feat["year"] or _track_year(t),
                "genres": _feat["genres"],
                "album": _feat["album"],
                "explicit": _feat["explicit"],
                "popularity": _feat["popularity"],
                "loudness": _feat["loudness"],
            }
        )
    logger.info("ym similar %s -> %d tracks", track_id, len(out))
    # Блок-лист мусора (Macarena и т.п.) — выкидываем из пула похожих.
    _before = len(out)
    out = [t for t in out if not _is_blocked(t)]
    if len(out) != _before:
        logger.info("blocklist similar: %d -> %d", _before, len(out))
    return out


async def rotor_tracks(
    track_id: str, limit: int = 30, queue_id: str | None = None
) -> tuple[list[dict], str | None]:
    """Бесконечный поток Rotor-сессии (то, что в проде даёт «Моя волна»).

    GET /rotor/station/track:<id>/tracks?settings2=True[&queue=<id>] —
    сессия Яндекса возвращает батч (sequence[]) личных рекомендаций.
    ВАЖНО: путь `station` (ед.ч.) — `stations` даёт 404 (проверено).
    queue_id — id трека, что играл перед этим (продолжение сессии → СВЕЖИЙ
    батч, а не тот же список); без queue — первый батч от станции.
    Возвращает (tracks, batch_id) — batch_id нужен для rotor_feedback().
    """
    if not track_id.startswith(_YM_ID_PREFIX):
        return [], None
    raw_id = track_id[len(_YM_ID_PREFIX):]
    params: dict = {"settings2": "True"}
    if queue_id:
        q = str(queue_id).replace(_YM_ID_PREFIX, "")
        if q:
            params["queue"] = q
    try:
        resp = await _get_client().get(
            f"{API_BASE}/rotor/station/track:{raw_id}/tracks",
            params=params,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ym rotor failed for %s: %s", track_id, e)
        return [], None
    if resp.status_code != 200:
        logger.warning("ym rotor HTTP %d for %s", resp.status_code, track_id)
        return [], None
    res = resp.json().get("result") or {}
    batch_id = res.get("batchId") or res.get("batch_id") or None
    seq = res.get("sequence") or []
    out: list[dict] = []
    for item in seq[:limit]:
        t = item.get("track") if isinstance(item, dict) else None
        if not t or not t.get("available"):
            continue
        tid = t.get("id")
        if tid is None:
            continue
        _feat = _track_features(t)
        out.append(
            {
                "id": f"{_YM_ID_PREFIX}{tid}",
                "title": t.get("title") or "",
                "artist": ", ".join(a.get("name", "") for a in (t.get("artists") or [])),
                "duration": int((t.get("durationMs") or 0) // 1000),
                "thumbnail": _cover_url(t.get("coverUri")),
                "channel": "",
                "url": f"https://music.yandex.ru/track/{tid}",
                "year": _feat["year"] or _track_year(t),
                "genres": _feat["genres"],
                "album": _feat["album"],
                "explicit": _feat["explicit"],
                "popularity": _feat["popularity"],
                "loudness": _feat["loudness"],
            }
        )
    logger.info(
        "ym rotor %s -> %d tracks (batch=%s)",
        track_id, len(out), bool(batch_id),
    )
    # Блок-лист мусора (Macarena и т.п.) — выкидываем из ротор-пула.
    _before = len(out)
    out = [t for t in out if not _is_blocked(t)]
    if len(out) != _before:
        logger.info("blocklist rotor: %d -> %d", _before, len(out))
    return out, batch_id


async def rotor_feedback(track_id: str, batch_id: str, kind: str = "trackStarted") -> bool:
    """Обратная связь Rotor-сессии: следующий батч станет СВЕЖИМ (другие треки).

    POST /rotor/station/track:<id>/feedback?batch-id=... {type, timestamp, trackId}.
    Проверено: trackStarted → 200 (это оно и даёт подстройку); вызывается при
    старте трека из рекомендаций. Молчаливый фолбэк при ошибке.
    """
    if not track_id.startswith(_YM_ID_PREFIX) or not batch_id:
        return False
    raw_id = track_id[len(_YM_ID_PREFIX):]
    try:
        resp = await _get_client().post(
            f"{API_BASE}/rotor/station/track:{raw_id}/feedback",
            params={"batch-id": str(batch_id)},
            json={
                "type": kind,
                "timestamp": int(time.time() * 1000),
                "trackId": str(track_id).replace(_YM_ID_PREFIX, ""),
            },
            timeout=10.0,
        )
        ok = resp.status_code == 200
        if not ok:
            logger.info("ym rotor feedback %s -> HTTP %d", kind, resp.status_code)
        return ok
    except Exception as e:  # noqa: BLE001
        logger.info("ym rotor feedback failed: %s", e)
        return False


# ------------------------------------------------------------ direct link --
def _build_direct_link(xml: str) -> str:
    """XML download-info → прямая ссылка на mp3 (живёт ~60с)."""

    def node(tag: str) -> str:
        m = re.search(f"<{tag}>(.*?)</{tag}>", xml)
        return m.group(1) if m else ""

    host, path, ts, s = node("host"), node("path"), node("ts"), node("s")
    if not (host and path and ts and s):
        raise ValueError("download-info XML incomplete")
    sign = hashlib.md5((_SIGN_SALT + path[1:] + s).encode("UTF-8")).hexdigest()
    return f"https://{host}/get-mp3/{sign}/{ts}{path}"


async def direct_link(track_id: str) -> str:
    """ym_<id> → свежая прямая ссылка на файл (через прокси для метаданных)."""
    bare = track_id[len(_YM_ID_PREFIX):]
    if not bare.isdigit():
        raise ValueError(f"bad yandex track id: {track_id!r}")
    resp = await _get_client().get(
        f"{API_BASE}/tracks/{bare}/download-info",
        params={"codec": "mp3", "quality": "high"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Yandex download-info HTTP {resp.status_code}")
    result = (resp.json().get("result") or [{}])[0]
    info_url = result.get("downloadInfoUrl")
    if not info_url:
        raise RuntimeError("Yandex: no downloadInfoUrl")
    r2 = await _get_client().get(info_url)
    if r2.status_code != 200:
        raise RuntimeError(f"Yandex download-info xml HTTP {r2.status_code}")
    return _build_direct_link(r2.text)


# ------------------------------------------------- cache (редкие сбросы) --
_link_cache: dict[str, tuple[str, float]] = {}
_LINK_TTL = 45.0  # < 60с — Яндекс отдаёт 410 по истечении


async def cached_direct_link(track_id: str, force: bool = False) -> str:
    now = time.time()
    if not force:
        hit = _link_cache.get(track_id)
        if hit and now - hit[1] < _LINK_TTL:
            return hit[0]
    url = await direct_link(track_id)
    _link_cache[track_id] = (url, now)
    if len(_link_cache) > 300:  # не раздувать
        for k in sorted(_link_cache, key=lambda k: _link_cache[k][1])[:100]:
            _link_cache.pop(k, None)
    return url


# ------------------------------------------------------------- stream side --
# Файлы storage.yandex.net качаются НАПРЯМУЮ (без прокси — проверено 206).
# Отдельный клиент: общий get_http_client() имеет local_address="::"
# (нужен YouTube CDN) и может не пройти до IPv4 storage Яндекса.
_stream_client: httpx.AsyncClient | None = None


def stream_client() -> httpx.AsyncClient:
    global _stream_client
    if _stream_client is None or _stream_client.is_closed:
        _stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=None),
            follow_redirects=False,
        )
    return _stream_client


async def close_stream() -> None:
    global _stream_client
    if _stream_client is not None and not _stream_client.is_closed:
        await _stream_client.aclose()
    _stream_client = None
