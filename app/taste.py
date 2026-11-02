"""Профиль вкуса (Фичи 1/2/4): любимые артисты + обратная связь.

- taste: любимые артисты → холодный старт рекомендаций (визард в APK).
- feedback: like/dislike/skip → фильтры выдачи (Яндекс rotor принимает
  только trackStarted/like/unlike — dislike/skip локально, проверено 400).

Хранение: data/taste.json, data/feedback.json (потокобезопасно, атомарно).
Один пользователь (personal server) — глобальное хранилище корректно.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("taste")

_DATA = Path(__file__).resolve().parent.parent / "data"
_LOCK = threading.Lock()

# Скип = сигнал «не сейчас/не это»: 1 скип убирает трек из РЕКОМЕНДАЦИЙ,
# но НЕ из точного поиска; 5+ скипов одного артиста → бан артиста в рекомендациях.
# 2026-11-01: 3 → 5 (Алекс: «просто послушал и переключил» легко накапливается,
# не должно вылетать навсегда). Гибкость даёт session-skip (понижение в сессии)
# и 👎 (мгновенный бан) — см. record(context=).
_SKIP_TRACK_FLOOR = 1
_SKIP_ARTIST_FLOOR = 5


def _load(name: str) -> dict:
    p = _DATA / name
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("taste load %s: %s", name, e)
    return {}


def _save(name: str, obj: dict) -> None:
    try:
        _DATA.mkdir(parents=True, exist_ok=True)
        tmp = _DATA / (name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=0), encoding="utf-8")
        os.replace(tmp, _DATA / name)
    except Exception as e:  # noqa: BLE001
        logger.warning("taste save %s: %s", name, e)


# ---------------------------------------------------------------- taste ----
def get_taste() -> list[str]:
    return list((_load("taste.json") or {}).get("artists") or [])


def set_taste(artists: list[str]) -> list[str]:
    """Сохранить любимые артисты (дедуп, cap 20, case-insensitive)."""
    seen: dict[str, str] = {}
    for a in artists:
        a = (a or "").strip()
        if not a:
            continue
        k = a.lower()
        if k not in seen:
            seen[k] = a
        if len(seen) >= 20:
            break
    out = list(seen.values())
    with _LOCK:
        _save("taste.json", {"artists": out, "updated": int(time.time())})
    logger.info("taste saved: %d artists", len(out))
    return out


# ------------------------------------------------------------- feedback ----
def record(kind: str, track_id: str, artist: str = "", progress: float = -1.0,
           genres: list[str] | None = None, context: str = "") -> None:
    """like | dislike | skip | completed для трека (track_id 'ym_...').

    ТЗ п.6.4 (2026-10-31): пропуск ≠ плохой трек автоматически.
      - skip progress<0.3 (или неизвестен) → ранний: +1.0 (сильный негатив)
      - skip 0.3..0.7 → средний: +0.5
      - skip >=0.7 → поздний: +0.25 (слабый негатив — почти дослушал)
      - completed → позитивный счётчик (для будущей модели, НЕ вкус артистов)
    Обратная совместимость: progress=-1 → прежнее поведение (+1).

    ТЗ п.3 (2026-10-31): дополнительно накапливается artist_scores —
    ВЕС артиста с меткой времени (для затухания и дифференциации
    «любимый vs слушал однажды vs надоел»). Один скип ≠ ненависть
    (пороги _SKIP_* не трогаем) — вес просто уходит в минус слабо.

    CONTEXT (2026-11-01, «слышал уже, хочу другую» vs «не нравится»):
      - context='manual' → ручное «следующая»: МЯГЧЕ. Прогресс ≥0.3 → вес 0.35
        (не 0.5): дослушал половицу и переключил = «хватит сейчас», не «плохо».
        Ранний (<0.3) — как всегда 1.0.
      - context='auto'   → быстрое автопереключение: прежняя шкала по прогрессу.
      - context=''        → обратная совместимость (прежнее поведение).
      Любой skip дополнительно пишет SESSION-сигнал артиста (см.
      _bump_artist_score source='session-skip') — понижение в ТЕКУЩЕЙ подборке
      без бана; не бан — это и есть «сейчас не хочу» vs «вообще не нравится».
    """
    if not track_id:
        return
    with _LOCK:
        fb = _load("feedback.json")
        now = int(time.time())
        a_key = (artist or "").strip().lower()[:60]
        delta = 0.0  # общий вес события (для artist_scores И genre_scores)
        if kind == "like":
            fb.setdefault("liked", {})[track_id] = now
            # лайк снимает старые скипы трека
            (fb.setdefault("skipped", {})).pop(track_id, None)
            delta = +2.0
            _bump_artist_score(fb, a_key, delta, now, source="player")
        elif kind == "unlike":
            # ТЗ п.8: unlike — отдельное событие (снятие лайка, только визуал
            # раньше). Снимаем лайк и возвращаем вес вниз, если лайк был.
            if fb.setdefault("liked", {}).pop(track_id, None) is not None:
                delta = -2.0
                _bump_artist_score(fb, a_key, delta, now, source="player-unlike")
            _save("feedback.json", fb)
            logger.info("feedback unlike %s (artist=%r)", track_id, artist[:30])
            return
        elif kind == "dislike":
            fb.setdefault("disliked", {})[track_id] = now
            delta = -2.5
            _bump_artist_score(fb, a_key, delta, now, source="player")
        elif kind == "completed":
            # дослушивание — положительный сигнал (ТЗ 6.4); вес +1
            cp = fb.setdefault("completed", {})
            cp[track_id] = cp.get(track_id, 0) + 1
            # «просто слушал» накапливается: 1 дослушивание = слабо,
            # 3+ = уже заметный плюс (любимый vs однажды слушал)
            delta = +0.35
            _bump_artist_score(fb, a_key, delta, now, source="player")
        elif kind == "skip":
            # вес скипа по проценту прослушивания (ТЗ 6.4)
            if progress < 0:
                weight = 1.0            # прогресс неизвестен → как раньше
            elif progress < 0.3:
                weight = 1.0            # ранний скип — сильный негатив
            elif progress < 0.7:
                # РУЧНОЕ «следующая» с прослушанной серединой → мягче (0.35):
                # «слышал уже, хочу другую» ≠ «плохой трек». Auto/пусто — 0.5.
                weight = 0.35 if context == "manual" else 0.5
            else:
                weight = 0.25           # поздний — почти дослушал, слабый сигнал
            sk = fb.setdefault("skipped", {})
            sk[track_id] = sk.get(track_id, 0) + weight
            if artist and weight >= 0.5:
                sa = fb.setdefault("skipped_artists", {})
                key = artist.strip().lower()
                sa[key] = sa.get(key, 0) + weight
            # SESSION-сигнал (2026-11-01): артист понижается в ТЕКУЩЕЙ сессии —
            # «сейчас не хочу» (без ожидания накопления до бан-порога). Живёт в
            # session_scores с БЫСТРЫМ полураспадом (2 дня) — артист возвращается
            # сам, без бана. Ранний ручной скип сильнее позднего. Это НЕ жёсткий
            # бан (он по _SKIP_ARTIST_FLOOR / 👎) — просто реже выпадает в
            # ближайших подборках. Итог = artist_scores + session_weights().
            if artist:
                sess_delta = -0.9 if progress < 0.3 else -0.5
                _bump_session_score(fb, a_key, sess_delta, now)
            delta = -weight
            _bump_artist_score(fb, a_key, delta, now, source="player")
        else:
            return
        # ТЗ п.3: предпочитаемые жанры (те же события, что и у артистов)
        if delta and genres:
            _bump_genre_scores(fb, genres, delta, now)
        _save("feedback.json", fb)
    logger.info("feedback %s %s (artist=%r progress=%.2f)", kind, track_id, artist[:30], progress)


def _bump_artist_score(fb: dict, a_key: str, delta: float, now: int,
                       source: str = "") -> None:
    """Накопление веса артиста (ТЗ п.3/п.8): score + метка времени.

    a_key пустой → пропуск (artist неизвестен — событие не копится).
    ТЗ п.8: каждое изменение имеет ИСТОЧНИК, ВРЕМЯ, ВЕС и журнал событий
    (events, cap 10) — основа для отмены/аудита («источник, время, вес и
    возможность отмены»).
    """
    if not a_key:
        return
    sc = fb.setdefault("artist_scores", {})
    e = sc.get(a_key) or {"score": 0.0, "ts": now, "events": []}
    e["score"] = round(float(e.get("score", 0.0)) + delta, 3)
    e["ts"] = now
    ev = e.setdefault("events", [])
    if isinstance(ev, list):
        ev.append({"ts": now, "delta": round(delta, 3), "src": source[:24]})
        del ev[:-10]  # cap: последние 10 событий (аудит + отмена)
    sc[a_key] = e


def _bump_session_score(fb: dict, a_key: str, delta: float, now: int) -> None:
    """Session-сигнал (2026-11-01): временное «сейчас не хочу артиста».

    Живёт в session_scores с полураспадом _SESSION_HALF_LIFE_DAYS (2 дня) —
    быстрее artist_scores (30 дней). Суммируется в artist_weights() при
    ранжировании: артист уходит из ТЕКУЩИХ подборок, но возвращается сам.
    Это гибкая альтернатива «убрать/не убирать» (Алекс: не тупой бинарь).
    """
    if not a_key:
        return
    ss = fb.setdefault("session_scores", {})
    e = ss.get(a_key) or {"score": 0.0, "ts": now}
    e["score"] = round(float(e.get("score", 0.0)) + delta, 3)
    e["ts"] = now
    ss[a_key] = e


def record_artist_preference(artist: str, delta: float,
                             source: str = "assistant") -> float:
    """ТЗ п.8: валидируемая операция профиля от ассистента (текстовая обратная

    связь: «этот исполнитель надоел»). Только через ВЕС (delta) с источником —
    БД напрямую LLM не трогает. Возвращает новый итоговый вес.
    """
    a_key = (artist or "").strip().lower()[:60]
    if not a_key or not delta:
        return 0.0
    with _LOCK:
        fb = _load("feedback.json")
        now = int(time.time())
        _bump_artist_score(fb, a_key, float(delta), now, source=source)
        _save("feedback.json", fb)
    w = artist_weights().get(a_key, 0.0)
    logger.info("assistant preference: %r delta=%+.2f -> weight=%.2f (src=%s)",
                artist[:40], delta, w, source)
    return w


def _bump_genre_scores(fb: dict, genres: list[str], delta: float, now: int) -> None:
    """Накопление весов ЖАНРОВ (ТЗ п.3.2: предпочитаемые жанры)."""
    gs = fb.setdefault("genre_scores", {})
    for g in genres[:5]:
        gk = str(g or "").strip().lower()[:40]
        if not gk:
            continue
        e = gs.get(gk) or {"score": 0.0, "ts": now}
        e["score"] = round(float(e.get("score", 0.0)) + delta, 3)
        e["ts"] = now
        gs[gk] = e


# Затухание (ТЗ п.3): полупериод 30 дней — старое событие влияет вдвое слабее
# каждые 30 суток. Профиль эволюционирует, а не копит вечно.
_DECAY_HALF_LIFE_DAYS = 30.0
# Session-сигнал (2026-11-01, «сейчас не хочу артиста»): БЫСТРЫЙ полураспад —
# за 2 дня влияние исчезает, артист возвращается сам. Это гибкая альтернатива
# «убрать/не убирать»: понижает в текущих подборках, но не навсегда.
_SESSION_HALF_LIFE_DAYS = 2.0


def _decayed_weights(scores: dict, half_life: float | None = None) -> dict[str, float]:
    """Общий расчёт: score+ts → вес с экспоненциальным затуханием.

    half_life — полупериод в днях; по умолчанию _DECAY_HALF_LIFE_DAYS (30).
    Для session_scores передаётся быстрый _SESSION_HALF_LIFE_DAYS (2).
    """
    hl = _DECAY_HALF_LIFE_DAYS if half_life is None else half_life
    out: dict[str, float] = {}
    now = time.time()
    for k, e in (scores or {}).items():
        if not k:
            continue
        try:
            score = float(e.get("score", 0.0))
            ts = float(e.get("ts", 0)) or now
        except (TypeError, ValueError):
            continue
        if score == 0.0:
            continue
        age_days = max(0.0, (now - ts) / 86400.0)
        decay = 0.5 ** (age_days / hl)
        w = score * decay
        # порог «шума»: |w| < 0.15 — не влияет на порядок
        if abs(w) >= 0.15:
            out[k] = round(w, 3)
    return out


def session_weights() -> dict[str, float]:
    """Session-сигнал «сейчас не хочу артиста» (2026-11-01).

    Полупериод 2 дня — быстрое понижение в ТЕКУЩИХ подборках, артист
    возвращается сам. Суммируется с artist_weights() в ранжере.
    """
    return _decayed_weights(_fb().get("session_scores"), half_life=_SESSION_HALF_LIFE_DAYS)


def artist_weights() -> dict[str, float]:
    """Веса артистов из обратной связи, с затуханием по времени (ТЗ п.3).

    Ключ — lowercase имя артиста, значение — итоговый вес [-inf..+inf]:
      +2.0 за лайк, +0.35 за каждое дослушивание, -2.5 за дизлайк,
      -(1.0/0.5/0.25) за скип (по прогрессу).
    + SESSION-сигнал (полупериод 2 дня): понижение «сейчас не хочу» —
    артист уходит из ближайших подборок, но возвращается, не бан.
    Пороги банов (5+ скипа / 👎) продолжают работать ОТДЕЛЬНО.
    Ручные taste-артисты (визард) не входят — они бинарно дают 1.0 в rank.
    """
    merged = dict(_decayed_weights(_fb().get("artist_scores")))
    for k, sw in session_weights().items():
        merged[k] = round(merged.get(k, 0.0) + sw, 3)
    return merged


def genre_weights() -> dict[str, float]:
    """ТЗ п.3.2: предпочитаемые жанры с затуханием (те же события)."""
    return _decayed_weights(_fb().get("genre_scores"))


def _fb() -> dict:
    return _load("feedback.json") or {}


def banned_track_ids(hard_only: bool = False) -> set[str]:
    """hard_only=True: только дизлайки (точный поиск); иначе + скипы."""
    fb = _fb()
    out = set(fb.get("disliked") or {})
    if not hard_only:
        for tid, cnt in (fb.get("skipped") or {}).items():
            if cnt >= _SKIP_TRACK_FLOOR:
                out.add(tid)
    return out


def banned_artists() -> set[str]:
    fb = _fb()
    return {a for a, c in (fb.get("skipped_artists") or {}).items()
            if c >= _SKIP_ARTIST_FLOOR}


def filter_tracks(tracks: list, hard_only: bool = False) -> list:
    """Выкинуть треки под запрет. hard_only (точный поиск) = только дизлайки.
    Работает и с dict, и с Pydantic TrackMeta (по .id/.artist)."""
    if not tracks:
        return tracks
    ban = banned_track_ids(hard_only=hard_only)
    ban_art = set() if hard_only else banned_artists()
    out = []
    dropped = 0
    for t in tracks:
        tid = t.get("id") if isinstance(t, dict) else getattr(t, "id", "")
        art = ((t.get("artist") if isinstance(t, dict) else getattr(t, "artist", "")) or "")
        if tid in ban or art.strip().lower() in ban_art:
            dropped += 1
            continue
        out.append(t)
    if dropped:
        logger.info("taste filter: %d -> %d (hard_only=%s, banned=%d)",
                    len(tracks), len(out), hard_only, len(ban) + len(ban_art))
    return out


def stats() -> dict:
    fb = _fb()
    return {
        "artists": get_taste(),
        "liked": len(fb.get("liked") or {}),
        "disliked": len(fb.get("disliked") or {}),
        "skipped": len(fb.get("skipped") or {}),
    }


def known_artists() -> set[str]:
    """Артисты, которых пользователь УЖЕ ЗНАЕТ (ТЗ п.12 «Открытия»).

    = вкус (визард/лайки) ∪ дизлайки-артисты ∪ часто скипаемые (>=порога).
    Используется режимом discover: кандидаты ВНЕ этого множества = новые.
    """
    fb = _fb()
    out: set[str] = set()
    for a in (get_taste() or []):
        s = a.strip().lower()
        if s:
            out.add(s)
    for k, _v in (fb.get("disliked_artists") or {}).items():
        if k:
            out.add(str(k).strip().lower())
    for k, c in (fb.get("skipped_artists") or {}).items():
        if k and (c or 0) >= _SKIP_TRACK_FLOOR:
            out.add(str(k).strip().lower())
    return out
