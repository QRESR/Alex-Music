"""Alice-like intent parsing for music search.

Turns a raw voice/text query into a structured intent the search route can
act on:

    "подборку групп крови"        -> intent=playlist, query="группа крови"
    "включи басту"                -> intent=play,      query="баста"
    "что послушать?"              -> intent=suggest
    "микс кино"                   -> intent=playlist, query="кино микс"
    "караоке версия"              -> intent=tag,       query=..., tag="караоке"
    "спокойная ночь"              -> intent=play,      query="спокойная ночь"

Design notes:
- Regex/rule based (no network, no LLM): fast, deterministic, offline-safe.
- Strips Russian request "noise" verbs so the residual is a clean query.
- Detects playlists/radio/mix semantics so the caller can ask YouTube for a
  *related* set (mix/playlist) instead of a literal title match.
- Detection of "already knows you" is heuristic: if the query names a
  playlist we hold, return it directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Request verbs / politeness that carry no search meaning.
_NOISE = (
    r"включи|включить|включим|запусти|запустить|поставь|поставить|"
    r"пошли|поиграй|поиграть|сиграй|сыграй|играй|играть|послушай|послушать|"
    r"давай|пожалуйста|мне|моя|мое|мой|какую|какие|какой|хочу|хочется|"
    r"дай|дай-ка|ну|а|please"
)

# Signals that we want a SET of tracks (playlist/radio/mix), not one title.
# Concrete SET nouns only — adjectives stay out (see note below).
_PLAYLIST_WORDS = (
    "подборк", "плейлист", "микс", "радио", "коллекц", "список",
    "тренд", "популярн", "лучш", "заряжен",  # concrete SET nouns
    "композиц", "релакс", "чарт", "хит", "сборник", "микстейп",
    # латиница (APK шлёт «playlist lullaby», «radio», «mix»)
    "playlist", "playlist", "mix", "radio", "set", "best of", "hits",
    # «по настроению / похожее / другое» = просим СВЯЗНЫЙ набор (как Алиса)
    "по настроению", "настроению", "похож", "другой", "другого", "другого исполнителя",
    "не этого", "подбери", "что-нибудь", "что нибудь",
    # "for <context>" combos read as a set request.
    "для тренировки", "для настроения", "для дороги", "для сна",
    "для бега", "для зала", "для работы", "для учёбы",
)
# NOTE: "танцев" removed from unconditional — it pairs with a genre adjective
# ("танцевальную зарубежную") which parse_intent now routes as a genre query.

# Mood / genre adjectives -> normalized to the noun form so YouTube gets a
# clean, search-friendly query ("веселую танцевальную зарубежную" -> "веселая
# танцевальная зарубежная музыка").
_ADJECTIVE_TO_GENRE = {
    "весел": "веселая", "весёл": "веселая",
    "танцевальн": "танцевальная", "танцевальную": "танцевальная",
    "релакс": "релакс", "расслаб": "релакс",
    "спокойн": "спокойная", "мягк": "мягкая", "тих": "тихая",
    "энергичн": "энергичная", "энергичную": "энергичная",
    "грустн": "грустная", "мрачн": "мрачная",
    "романтичн": "романтичная", "нежн": "нежная",
    "боев": "боевая", "эпичн": "эпическая",
}
# Set-noun suffixes we normalise onto the adjective ("танцевальную" ->
# "танцевальная") by matching the adjective root.
_SET_SUFFIXES = ("ую", "юю", "ую", "ая", "яя", "ое", "ее", "ые", "ие", "ую")

# Genre labels (kept as-is in the query; they are what YouTube matches on).
_GENRE_WORDS = (
    "зарубежн", "зарубежная", "русск", "русская", "зарубежную", "русскую",
    "зарубежного", "русского", "зарубежное", "русское",
    "поп", "рок", "рэп", "хип-хоп", "хип хоп", "реп", "джаз", "блюз",
    "диско", "хаус", "техно", "транс", "фонк", "инди", "альтернатив",
    "классическ", "опера", "шансон", "фанк", "соул", "кантри", "метал",
    "латино", "регги", "эмбиент", "lo-fi", "лоу-фай",
    # ФИКС «все жанры» (Алиса): электронная/танцевальная/романтическая и др.
    # иначе _is_pure_genre=False → LLM портит жанр отсебятиной («для тренировки»).
    "электрон", "танцевальн", "танцев", "романтическ", "романтик",
    "релакс", "расслаб", "спокойн", "мягк", "тихий", "чилл", "лойфай", "лоуфай",
    "бодр", "энергичн", "динамичн", "боев", "эпичн",
    "весел", "весёл", "грустн", "мрачн", "радостн", "меланхол",
    "панк", "эмок", "металкор", "фолк", "музыкальный",
)

# Artist vs title hints: "от <X>" / "исполняет <X>" / "<X> песни" -> that is
# the artist, not a title. We keep the whole phrase as the query (YouTube is
# good at "баста песни") but mark it so the caller can bias a related set.
_ARTIST_HINT = re.compile(r"\b(?:от|исполняет|исполнитель|группа|бэнд|дуэт)\s+")
# The word "музыка/песни/треки/композиции/штуки" alone signals a set request
# even with no other SET noun present ("музыку дай", "песни для дороги").
_MUSIC_SET = re.compile(r"\b(?:музык\w*|песн\w*|трек\w*|композиц\w*|штук\w*)\b")

# Intent classes.
INTENT_PLAY = "play"
INTENT_PLAYLIST = "playlist"
INTENT_SUGGEST = "suggest"


@dataclass
class Intent:
    intent: str = INTENT_PLAY
    query: str = ""
    is_playlist: bool = False
    matched_playlist: str | None = None  # name of a stored playlist, if any
    matched_playlist_id: str | None = None
    tags: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def wants_many(self) -> bool:
        """True when the caller should return several tracks / a set."""
        return self.intent == INTENT_PLAYLIST or self.is_playlist


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("ё", "е")
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def _strip_noise(t: str) -> str:
    # Remove request words as whole words, repeatedly.
    for _ in range(3):
        new = re.sub(rf"(?<!\w)(?:{_NOISE})(?!\w)", " ", t)
        new = re.sub(r"\s+", " ", new).strip()
        if new == t:
            break
        t = new
    return t


def _normalize_adjectives(t: str) -> str:
    """Rewrite mood adjectives to their noun form + append 'музыка' if the
    query is now only adjectives/genres (so YouTube gets a coherent query).

    "веселую танцевальную зарубежную" -> "веселая танцевальная зарубежная музыка"
    Leaves anything with a concrete noun alone (e.g. "спокойная ночь" keeps
    "ночь" and is not turned into a genre).
    """
    # Split into tokens; only touch tokens that are known adjective roots.
    tokens = t.split()
    out_tokens: list[str] = []
    changed = False
    for tok in tokens:
        bare = tok.strip("«»\"'.,!?-")
        matched = None
        for root, genre in _ADJECTIVE_TO_GENRE.items():
            if bare.startswith(root):
                # base already ends in the feminine noun ("ая"/"ая"); just use
                # it verbatim, stripping any dative/accusative suffix.
                matched = genre
                # keep a legit non-adjective tail (e.g. none normally) — genre
                # entries are already full noun forms, so no suffix merge.
                break
        if matched:
            out_tokens.append(matched)
            changed = True
            continue
        # Adjective suffix normalisation for words not in the explicit map:
        # "танцевальную"/"зарубежную" -> "танцевальная"/"зарубежная" (fem. nom).
        if bare.endswith(("ую", "юю")):
            out_tokens.append(bare[:-2] + "ая")
            changed = True
            continue
        if bare.endswith(("ий", "ый")):
            out_tokens.append(bare[:-2] + "ой")
            changed = True
            continue
        out_tokens.append(tok)
    if not changed:
        return t
    result = " ".join(out_tokens)
    # If after normalisation the query has no concrete noun left (only
    # adjectives/genres), append "музыка" so it reads as a genre request.
    has_concrete = any(
        w not in _GENRE_WORDS
        and not any(g in w for g in _GENRE_WORDS)
        and not any(w.startswith(r) for r in _ADJECTIVE_TO_GENRE)
        and w not in ("музыка", "музыку", "музыки", "музыке", "что-нибудь", "что-то")
        for w in result.split()
    )
    if not has_concrete:
        # Avoid a doubled word when the query already ends in "музыку/музыка".
        if not re.search(r"\bмузык\w*\b", result):
            result = f"{result} музыка"
    return result


def _has_genre(t: str) -> bool:
    return any(g in t for g in _GENRE_WORDS)


# Concrete song-title nouns: if present, the query is a SINGLE track ("спокойная
# ночь", "черный день"), NOT a genre/mood set request.
_TITLE_NOUNS = (
    "ночь", "день", "утро", "вечер", "дождь", "солнце", "луна", "море",
    "дорога", "дом", "детство", "мама", "любовь", "сердце", "слезы",
    "поцелуй", "прощание", "встреча", "вера", "надежда", "счастье",
    "гонка", "падение", "взлёт", "полет", "полёт", "тучи", "ветер",
)


def _is_title_query(t: str) -> bool:
    """True when the query names a concrete song title (keep as play intent)."""
    return any(f" {n}" in f" {t} " or t.endswith(n) for n in _TITLE_NOUNS)


def parse_intent(raw: str, stored_playlists: list[dict] | None = None) -> Intent:
    """Parse a user query into an Intent. Never raises."""
    text = _normalize(raw)
    out = Intent(raw=raw or "")

    if not text:
        out.intent = INTENT_SUGGEST
        return out

    # Suggestion / greeting intent.
    if re.fullmatch(
        r"(что|чего|чё)\s+(мне\s+)?(послушать|послушаем|включить|включим)|"
        r"(что|чего|чё)\s+послушать\??|"
        r"(музыку|что-нибудь|что-то)\s+(дай|поставь|включи)|"
        r"подскажи\s+музыку|сурприз|на\s+выбор",
        text,
    ):
        out.intent = INTENT_SUGGEST
        return out

    # Playlist / set signals.
    playlist_signal = any(w in text for w in _PLAYLIST_WORDS)

    # Match against stored playlists ("подборку X" / "X" == existing name).
    # Require a meaningful overlap (>=3 chars) so single-letter or stopword
    # queries never spuriously match a stored playlist name.
    if stored_playlists and len(text) >= 3:
        for pl in stored_playlists:
            name = _normalize(pl.get("name", ""))
            if not name or len(name) < 3:
                continue
            if name in text or (len(text) >= 3 and text in name):
                out.matched_playlist = pl.get("name")
                out.matched_playlist_id = pl.get("id")
                out.intent = INTENT_PLAYLIST
                out.is_playlist = True
                out.query = name
                return out

    # Try to detect an explicit "подборку/плейлист <X>" phrase -> query X.
    m = re.search(
        r"(?:подборк\w*|плейлист\w*|микс|радио|список)\s+(?:такую\s+|вот\s+|"
        r"мою\s+|называется\s+|кальку\s+)?[«\"]?(.+?)[»\"]?$",
        text,
    )
    if m and playlist_signal:
        q = _strip_noise(m.group(1))
        if q:
            out.query = q
            out.intent = INTENT_PLAYLIST
            out.is_playlist = True
            return out

    # General cleanup for a plain query.
    q = _strip_noise(text)
    # drop a trailing "пожалуйста"-style leftovers and quotes
    q = q.strip(" «»\"',.!?-")
    if not q:
        out.intent = INTENT_SUGGEST
        return out

    # Genre / mood query: a mood adjective or a genre word (or "для <ситуация>")
    # means the user wants a SET, not one track. Normalise the adjectives into a
    # clean genre query so YouTube returns a coherent mix.
    #   "веселую танцевальную зарубежную" -> playlist, query "веселая танцевальная зарубежная музыка"
    # But a concrete song title ("спокойная ночь") stays a single play.
    is_genre_query = (
        (_has_genre(q) or any(q.startswith(r) or f" {r}" in q for r in _ADJECTIVE_TO_GENRE) or " для " in f" {q} ")
        and not _is_title_query(q)
        and not _ARTIST_HINT.search(q)
    )
    if is_genre_query:
        norm = _normalize_adjectives(q)
        out.query = norm
        out.intent = INTENT_PLAYLIST
        out.is_playlist = True
        return out

    # "музыку/песни <X>" or bare "музыку дай" -> a set request.
    if _MUSIC_SET.search(q):
        norm = _normalize_adjectives(q)
        # If normalisation left nothing usable (e.g. "музыку дай" -> "" after
        # noise strip), fall back to suggest rather than an empty set query.
        if not norm.strip():
            out.intent = INTENT_SUGGEST
            return out
        # If it still has a concrete title after cleanup (e.g. "песни басты"),
        # keep it as playlist (set of that artist's songs).
        out.query = norm
        out.intent = INTENT_PLAYLIST
        out.is_playlist = True
        return out

    out.query = q
    if playlist_signal:
        out.intent = INTENT_PLAYLIST
        out.is_playlist = True
    else:
        out.intent = INTENT_PLAY
    return out
