"""Audio embeddings: архитектура «похожесть по звучанию» (ТЗ п.4, шаг 6).

Реализация выбрана исследованием 2026-10-31 (research/audio-embeddings.md):
  model = laion/clap-htsat-unfused (Apache-2.0), dim=512, ~5.5 с/трек (batch=6),
  RAM 650-840 МБ, детерминизм cos=1.000 (фикс. окна 10 с — обход rand_trunc).

Принципы (ТЗ п.4.3):
  - ЛЕНИВЫЙ import torch — при рестарте сервиса модель НЕ грузится;
    тяжёлые пакеты поднимает только первый реальный вызов.
  - Кэш в SQLite (data/embeddings.db): трек НЕ анализируется повторно.
  - Не скачиваем музыку целиком: получаем байты ПОТОКОМ (range) только
    для треков, которых реально касается запрос.
  - Отказ модели/недоступность → БЕЗ эмбеддинга (fallback на метаданные);
    никаких выдуманных значений (ТЗ п.19).

API:
    embedding_for(track_id, audio_bytes) -> list[float] | None   # кэш + compute
    get_text_embedding(text) -> list[float] | None               # кэш текстов
    cosine(a, b) -> float
    nearest(vec, candidates: dict[str, list[float]], k) -> list[(id, cos)]
    stats() -> dict
"""
from __future__ import annotations

import io
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("embeddings")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.db"
MODEL_ID = "laion/clap-htsat-unfused"
LOCK = threading.Lock()          # main: сериализация записи кэша/инференса (2 vCPU)
_MODEL_LOCK = threading.Lock()   # отдельная: ленивая загрузка модели (БЕЗ вложенности в main)
_MODEL = _PROC = _AUDIO_KW = None  # singleton, ленивая загрузка

DDL = """
CREATE TABLE IF NOT EXISTS track_embeddings (
    track_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',  -- sha256 аудио: ловит ИЗМЕНИВШИЕСЯ треки (ТЗ п.6)
    embedding BLOB NOT NULL,      -- float32[512] = 2 КБ/трек
    computed_at INTEGER NOT NULL
);
-- Метаданные для CLAP-кандидатов (ТЗ similar 3.2): nearest_tracks() должен
-- вернуть artist/title/year, а эмбеддинг-кэш раньше хранил только вектор.
-- Отдельная таблица: не ломает INSERT эмбеддингов, заполняется из пула/backfill.
CREATE TABLE IF NOT EXISTS track_meta_cache (
    track_id TEXT PRIMARY KEY,
    artist TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    year INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS text_embeddings (
    text_hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL,
    computed_at INTEGER NOT NULL
);
"""


# ─── sqlite ────────────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(DDL)
        # миграция старых БД (патч 2, 2026-10-31): у text_embeddings
        # появилась колонка model (версионирование, ТЗ п.5);
        # у track_embeddings — content_hash (обнаружение изменившихся, ТЗ п.6).
        cols = [r[1] for r in c.execute("PRAGMA table_info(text_embeddings)")]
        if cols and "model" not in cols:
            c.execute("ALTER TABLE text_embeddings ADD COLUMN model TEXT NOT NULL DEFAULT ''")
            logger.info("migrate: text_embeddings +model (версионирование)")
        cols_t = [r[1] for r in c.execute("PRAGMA table_info(track_embeddings)")]
        if cols_t and "content_hash" not in cols_t:
            c.execute("ALTER TABLE track_embeddings ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
            logger.info("migrate: track_embeddings +content_hash (обнаружение изменений)")


def _load_vec(blob: bytes) -> list[float]:
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32).tolist()


def stats() -> dict:
    try:
        with _conn() as c:
            n = c.execute("SELECT COUNT(*) FROM track_embeddings").fetchone()[0]
            nt = c.execute("SELECT COUNT(*) FROM text_embeddings").fetchone()[0]
        return {"model": MODEL_ID, "tracks": n, "texts": nt, "loaded": _MODEL is not None}
    except Exception as e:  # noqa: BLE001
        return {"model": MODEL_ID, "error": str(e)}


# ─── ленивая модель (первый вызов поднимает torch) ────────────────────────
def _ensure_model():
    global _MODEL, _PROC, _AUDIO_KW
    if _MODEL is not None:
        return _MODEL, _PROC
    # _MODEL_LOCK (НЕ общий LOCK!): main уже может держать LOCK внутри
    # embedding_for/get_text_embedding — повторный захват того же
    # threading.Lock = вечный deadlock (подтверждено тестом 2026-10-31).
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _PROC
        import inspect
        import torch
        from transformers import ClapModel, ClapProcessor

        torch.set_num_threads(2)
        _PROC = ClapProcessor.from_pretrained(MODEL_ID)
        _MODEL = ClapModel.from_pretrained(MODEL_ID).eval()
        _AUDIO_KW = (
            "audio" if "audio" in inspect.signature(_PROC.__call__).parameters
            else "audios"
        )
        logger.info("CLAP model loaded (lazy): %s", MODEL_ID)
    return _MODEL, _PROC


def _compute_audio(audio_bytes: bytes) -> list[float]:
    """Один эмбеддинг трека: фикс. окна 10 с → mean-pool → L2 (детерминизм)."""
    import numpy as np
    import soundfile as sf
    import soxr
    import torch

    SR, WINDOW_S, BATCH, MIN_SEG_S = 48000, 10.0, 6, 1.0
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    model, proc = _ensure_model()
    win = int(WINDOW_S * SR)
    segs = [wav[i: i + win] for i in range(0, len(wav), win)]
    segs = [s for s in segs if len(s) >= MIN_SEG_S * SR] or [wav]
    parts = []
    for i in range(0, len(segs), BATCH):
        inputs = proc(**{_AUDIO_KW: segs[i: i + BATCH]}, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            out = model.get_audio_features(**inputs)
            if not torch.is_tensor(out):
                out = out.pooler_output
            parts.append(torch.nn.functional.normalize(out.float(), dim=-1))
    vec = torch.nn.functional.normalize(torch.cat(parts).mean(dim=0), dim=-1)
    return vec.squeeze(0).tolist()


def _compute_text(text: str) -> list[float]:
    import torch

    model, proc = _ensure_model()
    inputs = proc(text=[text], return_tensors="pt")
    with torch.inference_mode():
        out = model.get_text_features(**inputs)
        if not torch.is_tensor(out):
            out = out.pooler_output
        vec = torch.nn.functional.normalize(out.float(), dim=-1)
    return vec.squeeze(0).tolist()


# ─── публичное API с кэшем ─────────────────────────────────────────────────
def embedding_for(
    track_id: str, audio_bytes: bytes | None = None, *, compute: bool = True
) -> list[float] | None:
    """Эмбеддинг трека: кэш → (если дали байты) compute → None (нет данных).

    compute=False — РЕЖИМ ПОЛЬЗОВАТЕЛЬСКОГО ЗАПРОСА (ТЗ п.4.4/п.7):
      ТОЛЬКО чтение кэша, БЕЗ запуска инференса (массовый анализ аудио в
      запросе запрещён — тяжело для 2 vCPU). Расчёт делает ФОНОВОЙ бэтчфилл.
      None = пока не посчитано → вызывающий остаётся на метаданных.

    НИКОГДА не бросает: сбой модели = None (вызывающий код остаётся на
    метаданных — ТЗ п.4.3 «не выдавать метаданные за анализ звучания»).
    """
    if not track_id:
        return None
    _ch = _content_hash(audio_bytes) if audio_bytes else ""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT embedding, content_hash FROM track_embeddings WHERE track_id=? AND model=?",
                (track_id, MODEL_ID),
            ).fetchone()
        if row:
            _emb_blob, _stored_hash = row[0], (row[1] or "")
            # ТЗ п.6: если передано аудио и его хэш РАСХОДИТСЯ с кэшем —
            # трек ИЗМЕНИЛСЯ → кэш устарел, пересчитываем (не отдаём старый).
            if _ch and _stored_hash and _ch != _stored_hash:
                logger.info("embedding stale (content changed): %s — пересчёт", track_id)
            else:
                return _load_vec(_emb_blob)
    except Exception as e:  # noqa: BLE001
        logger.warning("embeddings cache read failed: %s", e)
    if not audio_bytes or not compute:
        # ТЗ п.4.4: в пользовательском запросе инференс НЕ запускаем
        # (compute=False) — кэш либо есть, либо честно None (фон доложит).
        return None
    try:
        with LOCK:                       # serialize forward (2 vCPU)
            vec = _compute_audio(audio_bytes)
        import numpy as np
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO track_embeddings VALUES (?,?,?,?,?,?)",
                (track_id, MODEL_ID, len(vec), _ch, blob, int(time.time())),
            )
        logger.info("embedding computed+cached for %s (dim=%d)", track_id, len(vec))
        return vec
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding compute failed for %s: %s", track_id, e)
        return None


def _content_hash(audio_bytes: bytes) -> str:
    """sha256 аудио — для обнаружения ИЗМЕНИВШИХСЯ треков (ТЗ п.6)."""
    import hashlib
    return hashlib.sha256(audio_bytes).hexdigest()[:32] if audio_bytes else ""


def is_stale(track_id: str, audio_bytes: bytes) -> bool:
    """True = кэш устарел (нет записи ИЛИ хэш аудио разошёлся).

    Используется backfill'ом, чтобы пересчитать только новые/изменившиеся.
    """
    if not track_id or not audio_bytes:
        return False
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT content_hash FROM track_embeddings WHERE track_id=? AND model=?",
                (track_id, MODEL_ID),
            ).fetchone()
        if row is None:
            return True                      # ещё не посчитано
        stored = row[0] or ""
        if not stored:
            return False                     # старая запись без хэша → считаем валидной (не пересчитываем зря)
        return stored != _content_hash(audio_bytes)
    except Exception:  # noqa: BLE001
        return False


def get_text_embedding(text: str, *, compute: bool = True) -> list[float] | None:
    """Эмбеддинг текстового запроса (кэш по hash) — для «похожее по звучанию описанию».

    compute=False — режим пользовательского запроса (ТЗ п.4.4): только кэш,
    инференс текста не запускаем.
    """
    import hashlib

    t = (text or "").strip()
    if not t:
        return None
    th = hashlib.sha1(t.lower().encode("utf-8")).hexdigest()
    try:
        with _conn() as c:
            # версионирование (ТЗ п.5): кэш только от ТЕКУЩЕЙ модели,
            # иначе при смене MODEL_ID текстовый вектор старой модели
            # ошибочно применится к новому пространству.
            row = c.execute(
                "SELECT embedding FROM text_embeddings WHERE text_hash=? AND model=?",
                (th, MODEL_ID),
            ).fetchone()
        if row:
            return _load_vec(row[0])
    except Exception as e:  # noqa: BLE001
        logger.warning("text emb cache failed: %s", e)
    if not compute:
        # ТЗ п.4.4: в пользовательском запросе текст-инференс не запускаем
        return None
    try:
        with LOCK:
            vec = _compute_text(t)
        import numpy as np
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO text_embeddings VALUES (?,?,?,?,?)",
                (th, t, MODEL_ID, blob, int(time.time())),
            )
        return vec
    except Exception as e:  # noqa: BLE001
        logger.warning("text embedding failed: %s", e)
        return None


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float((np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0)
    return float(va @ vb) / denom


def audio_similarities(
    track_ids: list[str], reference: list[float]
) -> dict[str, float]:
    """Пакетная похожесть кандидатов к референсу — ТОЛЬКО чтение кэша (ТЗ п.4.4/п.7).

    - НЕ запускает инференс (только SELECT из SQLite; треков без эмбеддинга
      просто нет в ответе → их audio_sim=0 в ранжере, работают метаданные);
    - ОДИН SQL-запрос на список (не по одному) — быстро даже на 30 кандидатах;
    - возвращает {track_id: cos 0..1} — это признак для recommender.rank(),
      НЕ итоговый порядок (сам список кандидатов не меняется).
    """
    if not reference or not track_ids:
        return {}
    out: dict[str, float] = {}
    try:
        import numpy as np
        ref = np.asarray(reference, dtype=np.float32)
        ref = ref / (np.linalg.norm(ref) or 1.0)
        qmarks = ",".join("?" * len(track_ids))
        with _conn() as c:
            rows = c.execute(
                f"SELECT track_id, embedding FROM track_embeddings "
                f"WHERE model=? AND track_id IN ({qmarks})",
                [MODEL_ID, *track_ids],
            ).fetchall()
        for tid, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            out[tid] = float(max(-1.0, min(1.0, v @ ref)))
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("audio_similarities failed: %s", e)
        return {}


def nearest(
    vec: list[float], candidates: dict[str, list[float]], k: int = 10
) -> list[tuple[str, float]]:
    """Топ-k кандидатов по косинусу (in-memory numpy matmul)."""
    import numpy as np
    if not vec or not candidates:
        return []
    ids = list(candidates.keys())
    mat = np.asarray([candidates[i] for i in ids], dtype=np.float32)
    q = np.asarray(vec, dtype=np.float32)
    sims = mat @ (q / (np.linalg.norm(q) or 1.0))
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def put_meta(rows: list[dict]) -> int:
    """Метаданные CLAP-кандидатов (id/artist/title/year) в track_meta_cache."""
    clean = [
        (r["id"], (r.get("artist") or "")[:200], (r.get("title") or "")[:300],
         int(r.get("year") or 0))
        for r in rows if r.get("id")
    ]
    if not clean:
        return 0
    with _conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO track_meta_cache (track_id, artist, title, year) "
            "VALUES (?,?,?,?)", clean,
        )
    return len(clean)


def nearest_tracks(reference: list[float], k: int = 20) -> list[dict]:
    """CLAP audio-to-audio источник (ТЗ similar 3.2): топ-k кэшированных
    треков, похожих на референс-эмбеддинг. ТОЛЬКО чтение SQLite — без
    инференса, без сети, без загрузки модели. Без референса → [].
    Возвращает метаданные (artist/title/year) из track_meta_cache; треки
    без метаданных пропускаются (не выдумываем характеристики, ТЗ п.4).
    """
    if not reference:
        return []
    try:
        import numpy as np
        with _conn() as c:
            rows = c.execute(
                "SELECT track_id, embedding FROM track_embeddings WHERE model=?",
                (MODEL_ID,),
            ).fetchall()
            if not rows:
                return []
            ids = [r[0] for r in rows]
            mat = np.asarray(
                [np.frombuffer(r[1], dtype=np.float32) for r in rows],
                dtype=np.float32,
            )
            meta = dict(
                (tid, (a, t, y)) for tid, a, t, y in c.execute(
                    "SELECT track_id, artist, title, year FROM track_meta_cache"
                )
            )
        q = np.asarray(reference, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        # нормировка строк: НЕ через `or` (массив!) — нулевые строки делим на 1
        _norms = np.linalg.norm(mat, axis=1, keepdims=True)
        _norms[_norms == 0] = 1.0
        mat = mat / _norms
        sims = mat @ q
        order = np.argsort(-sims)[: max(k * 2, k)]   # добор: часть без меты
        out: list[dict] = []
        for i in order:
            tid = ids[int(i)]
            if tid not in meta:
                continue
            artist, title, year = meta[tid]
            if not title:
                continue
            out.append({
                "id": tid, "artist": artist, "title": title, "year": int(year),
                "sim": float(sims[int(i)]),
            })
            if len(out) >= k:
                break
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("nearest_tracks failed: %s", e)
        return []


def warmup() -> None:
    """Фоновая прогревка модели (вызывать только если включён предрасчёт)."""
    try:
        _ensure_model()
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding warmup failed: %s", e)
