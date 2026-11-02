#!/usr/bin/env python3
"""ИСПЫТАНИЯ CLAP (ТЗ п.8+9) — отдельные прогоны, не смешивать с eval поиска.

Режимы:
  --prepare N   предрасчёт эмбеддингов N уникальных ym_-треков из eval_v5.json
                (пул для audio-испытаний; resume по кэшу, nice, throttle)
  --text        TEXT-TO-AUDIO: 100 текстовых запросов
                - latency text-эмбеддинга (кэш vs инференс)
                - discrimination gap: cos(text, топ-5 обычного поиска) vs
                  cos(text, случайные 5 из другого запроса) — понимает ли CLAP
                - artist@5/genre@5 у CLAP-топа против топа обычного поиска
                  («сравнение с поиском по исполнителю и жанру», п.8)
  --audio       AUDIO-TO-AUDIO: 100 исходных композиций
                - для каждого: топ-5 по косинусу из остальных пула
                - artist@5/genre@5 против СЛУЧАЙНОЙ выборки (baseline)
                - разнообразие (uniq артистов@10), latency на трек, RAM
  Записывает research/eval_clap_results.json (сводка) — это и есть материал
  для п.9 «сравнить качество до/после».

Нагрузка: nice(10), ОДИН инференс за раз. НЕ запускать во время eval_vN!
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
EVAL_V5 = Path("/root/.hermes/profiles/gera/cache/scratch/eval/eval_v5.json")
POOL = ROOT / "research" / "eval_pool.json"   # build_pool.py: top с id
OUT = ROOT / "research" / "eval_clap_results.json"
random.seed(42)   # воспроизводимость выборок


def _load_eval_pool() -> list[dict]:
    """Треки топа обычного поиска С track_id (для эмбеддингов).

    eval JSON хранит top без id — поэтому пул собирается отдельным
    build_pool.py по тем же запросам (исходник — eval_v6).
    """
    if POOL.exists():
        d = json.loads(POOL.read_text())
        out: dict[str, dict] = {}
        for r in d:
            for t in (r.get("top") or []):
                tid = t.get("id")
                if tid and tid.startswith("ym_"):
                    out.setdefault(tid, t)
        if out:
            return list(out.values())
    # fallback: старый пул (без id — эмбеддинги не соберёшь, только текст-часть)
    d = json.loads(EVAL_V5.read_text())
    tracks: dict[str, dict] = {}
    for r in d["results"]:
        for t in (r.get("top") or []):
            tid = t.get("id")
            if tid and tid.startswith("ym_"):
                tracks.setdefault(tid, t)
    return list(tracks.values())


def _load_pool_rows() -> list[dict]:
    """Строки {query, type, top} с id — для baseline /search-топа в --text."""
    if POOL.exists():
        return json.loads(POOL.read_text())
    return []


def _queries(limit: int = 100) -> list[tuple[str, str]]:
    """100 текстовых запросов — из ПУЛА (те же, что у baseline)."""
    rows = _load_pool_rows()
    if rows:
        return [(r["query"], r.get("type") or "mix") for r in rows][:limit]
    d = json.loads(EVAL_V5.read_text())
    out = [(r["query"], r.get("type") or "mix") for r in d["results"] if "error" not in r]
    return out[:limit]


async def prepare(n: int) -> None:
    """Предрасчёт N треков из eval-пула (resume по кэшу)."""
    try:
        import os
        os.nice(10)
    except Exception:  # noqa: BLE001
        pass
    from app import embeddings as E
    from app import ym_client

    E.init_db()
    pool = _load_eval_pool()
    ids = [t["id"] for t in pool][:n]
    done = skipped = failed = 0
    t0 = time.time()
    for tid in ids:
        if E.embedding_for(tid, compute=False) is not None:
            skipped += 1
            continue
        try:
            url = await ym_client.cached_direct_link(tid)
            import httpx
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as hc:
                r = await hc.get(url)
                r.raise_for_status()
            vec = await asyncio.to_thread(E.embedding_for, tid, r.content, compute=True)
            done += 1 if vec else 0
            failed += 0 if vec else 1
            if done % 10 == 0:
                print(f"  prepared {done} ({skipped} cached, {failed} failed) "
                      f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {tid}: {e}", flush=True)
        await asyncio.sleep(1.5)
    print(f"PREPARE done={done} cached={skipped} failed={failed} "
          f"total={time.time()-t0:.0f}s stats={E.stats()}")


_DESC_WORDS = (
    "для дороги", "спокойн", "энергич", "бодр", "вечер", "утр", "ноч",
    "для работы", "настроени", "грустн", "весел", "драйв", "мощн",
    "лёгк", "легк", "спи", "тренаж", "заряд", "депресс", "романт", "поспокойнее",
)


def _is_descriptive(q: str) -> bool:
    """Описательный запрос («для дороги», «спокойная») — целевой для text-to-audio.

    Артистные («поставь эминема») CLAP по определению не знает (слеп к именам)
    — их считаем отдельно: там сравниваем поиск по артисту vs CLAP (п.8).
    """
    ql = q.lower()
    return any(w in ql for w in _DESC_WORDS)


def run_text() -> dict:
    """TEXT-TO-AUDIO: 100 запросов → метрики понимания звучания текстом."""
    import numpy as np
    from app import embeddings as E

    E.init_db()
    pool = _load_eval_pool()
    pool_ids = [t["id"] for t in pool]
    # эмбеддинги пула — только кэш (п.4.4)
    import sqlite3
    with sqlite3.connect(E.DB_PATH) as c:
        cached = {
            tid: np.frombuffer(b, dtype=np.float32)
            for tid, b in c.execute(
                "SELECT track_id, embedding FROM track_embeddings WHERE model=?",
                (E.MODEL_ID,),
            )
        }
    if not cached:
        return {"error": "кэш пуст — сначала --prepare"}
    usable = [t for t in pool if t["id"] in cached]
    print(f"text-испытание: пул {len(cached)} эмбеддингов из {len(pool)} треков")

    queries = _queries(100)
    lat_text, clap_artist_hit, base_artist_hit = [], 0, 0
    gaps_desc, overlaps, n_desc = [], [], 0
    for q, _typ in queries:
        t0 = time.perf_counter()
        ref = E.get_text_embedding(q, compute=False)   # кэш-only!
        dt = (time.perf_counter() - t0) * 1000
        if ref is None:
            # текста нет в кэше → считаем ОДИН РАЗ сюда (испытание, не прод-запрос)
            t0 = time.perf_counter()
            ref = E.get_text_embedding(q, compute=True)
            dt = (time.perf_counter() - t0) * 1000
            lat_text.append(dt)
        if ref is None:
            continue
        refv = np.asarray(ref, dtype=np.float32)
        refv = refv / (np.linalg.norm(refv) or 1.0)
        sims = {tid: float(refv @ v / (np.linalg.norm(v) or 1.0)) for tid, v in cached.items()}
        ranked = sorted(sims.items(), key=lambda x: -x[1])
        qwords = [w for w in q.lower().replace("ё", "е").split() if len(w) >= 4]
        by_id = {t["id"]: t for t in pool}
        clap_top5 = [by_id.get(tid) for tid, _ in ranked[:5]]
        clap_top5 = [t for t in clap_top5 if t]
        if clap_top5 and qwords:
            hit = sum(1 for t in clap_top5
                      if any(w in (t.get("artist") or "").lower() for w in qwords))
            clap_artist_hit += 1 if hit else 0
        # baseline: топ-5 обычного поиска (те же кандидаты, порядок eval'а)
        eval_top5 = [t for t in _search_top_for(q) if t["id"] in cached][:5]
        if eval_top5 and qwords:
            hit = sum(1 for t in eval_top5
                      if any(w in (t.get("artist") or "").lower() for w in qwords))
            base_artist_hit += 1 if hit else 0
        # overlap@5: сколько из CLAP-топ-5 есть в топе поиска (согласованность)
        if clap_top5 and eval_top5:
            cids = {t["id"] for t in clap_top5}
            eids = {t["id"] for t in eval_top5}
            overlaps.append(len(cids & eids) / 5.0)
        # discrimination gap — ТОЛЬКО по описательным (цель text-to-audio):
        # CLAP должен ставить топ-поиска выше случайных ПО ЗВУЧАНИЮ.
        if eval_top5 and _is_descriptive(q):
            n_desc += 1
            rel = [sims.get(t["id"], 0) for t in eval_top5]
            others = random.sample([tid for tid in sims if tid not in
                                    {t["id"] for t in eval_top5}], k=5)
            rnd = [sims.get(tid, 0) for tid in others]
            if rel and rnd:
                gaps_desc.append(statistics.mean(rel) - statistics.mean(rnd))

    n = len(queries)
    res = {
        "mode": "text",
        "queries": n,
        "pool": len(cached),
        "descriptive_queries": n_desc,
        "latency_text_ms_p50": round(statistics.median(lat_text), 1) if lat_text else None,
        "latency_text_ms_p95": round(sorted(lat_text)[int(len(lat_text) * 0.95)], 1) if len(lat_text) > 1 else None,
        "gap_desc_mean": round(statistics.mean(gaps_desc), 4) if gaps_desc else None,
        "clap_search_overlap@5": round(statistics.mean(overlaps), 3) if overlaps else None,
        "clap_artist_hit_rate": round(clap_artist_hit / n, 3),
        "search_artist_hit_rate": round(base_artist_hit / n, 3),
        "note": "gap>0 = CLAP понимает текст (релевантные выше случайных); "
                "hit_rate сравнивается с обычным поиском (п.8)",
    }
    return res


_search_top_cache: dict[str, list[dict]] = {}


def _search_top_for(q: str) -> list[dict]:
    """Baseline: топ обычного поиска по запросу (из пула с id)."""
    if not _search_top_cache:
        for r in _load_pool_rows():
            _search_top_cache[r["query"]] = r.get("top") or []
    return _search_top_cache.get(q, [])


def run_audio() -> dict:
    """AUDIO-TO-AUDIO: 100 треков → похожесть, сравнение со случайным baseline."""
    import numpy as np
    from app import embeddings as E

    E.init_db()
    pool = _load_eval_pool()
    import sqlite3
    with sqlite3.connect(E.DB_PATH) as c:
        cached = {
            tid: np.frombuffer(b, dtype=np.float32)
            for tid, b in c.execute(
                "SELECT track_id, embedding FROM track_embeddings WHERE model=?",
                (E.MODEL_ID,),
            )
        }
    if len(cached) < 20:
        return {"error": f"нужно ≥20 эмбеддингов, есть {len(cached)} — сначала --prepare"}
    by_id = {t["id"]: t for t in pool}
    seeds = [tid for tid in cached if tid in by_id][:100]
    lat, artist_hits, rnd_hits, uniques = [], 0, 0, []
    ids = list(cached.keys())
    for sid in seeds:
        t0 = time.perf_counter()
        v = cached[sid]
        v = v / (np.linalg.norm(v) or 1.0)
        sims = {tid: float(v @ w / (np.linalg.norm(w) or 1.0))
                for tid, w in cached.items() if tid != sid}
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        top5 = sorted(sims.items(), key=lambda x: -x[1])[:5]
        seed_artist = (by_id[sid].get("artist") or "").lower()
        # метрика: артист совпадает со seed (похожий исполнитель = осмысленный топ)
        for tid, _s in top5:
            a = (by_id.get(tid, {}).get("artist") or "").lower()
            if a and seed_artist and (a in seed_artist or seed_artist in a):
                artist_hits += 1
        # baseline: 5 случайных
        for tid in random.sample([i for i in ids if i != sid], k=5):
            a = (by_id.get(tid, {}).get("artist") or "").lower()
            if a and seed_artist and (a in seed_artist or seed_artist in a):
                rnd_hits += 1
        uniques.append(len({(by_id.get(tid, {}).get("artist") or "") for tid, _ in top5}))

    n = len(seeds)
    # Laplace-сглаживание: при rnd=0 enrichment не должен быть null (п.8)
    enrichment = round((artist_hits + 0.5) / (rnd_hits + 0.5), 2)
    res = {
        "mode": "audio",
        "tracks": n,
        "clap_same_artist@5_total": artist_hits,
        "random_same_artist@5_total": rnd_hits,
        "clap_enrichment_x": enrichment,
        "diversity_uniq_artists@5_mean": round(statistics.mean(uniques), 2) if uniques else None,
        "latency_cos_ms_p50": round(statistics.median(lat), 2) if lat else None,
        "latency_cos_ms_p95": round(sorted(lat)[int(len(lat) * 0.95)], 2) if len(lat) > 1 else None,
        "note": "enrichment_x = во сколько раз CLAP-топ чаще находит того же "
                "артиста, чем случайная выборка (п.8: сравнение с baseline)",
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", type=int, default=0, help="предрасчёт N треков из пула")
    ap.add_argument("--text", action="store_true", help="text-to-audio испытание")
    ap.add_argument("--audio", action="store_true", help="audio-to-audio испытание")
    args = ap.parse_args()

    results: dict = {}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except Exception:  # noqa: BLE001
            results = {}

    if args.prepare:
        asyncio.run(prepare(args.prepare))
    if args.text:
        t0 = time.time()
        r = run_text()
        r["wall_s"] = round(time.time() - t0, 1)
        results["text"] = r
        print(json.dumps(r, ensure_ascii=False, indent=1))
    if args.audio:
        t0 = time.time()
        r = run_audio()
        r["wall_s"] = round(time.time() - t0, 1)
        results["audio"] = r
        print(json.dumps(r, ensure_ascii=False, indent=1))

    if results:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
