#!/usr/bin/env python3
"""Бенч режима similar (ТЗ п.2/п.9): старая vs новая реализация.

Метрики на ВЫДАЧЕ (снаружи, честно):
  n, uniq@5/10/20, share_seed (доля seed-артиста), dup (повторы id),
  covers (каверы/версии в title), time_s, errors,
  clap_mean/cov (средний косинус к seed и покрытие — через embeddings-кэш).

Использование:
  python3 research/similar_bench.py --mode old --out research/similar_bench_old.json
  python3 research/similar_bench.py --mode new --out research/similar_bench_new.json
  python3 research/similar_bench.py --compare old new
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8011"
POOL = ROOT / "research" / "eval_pool.json"


def seeds(n: int = 40) -> list[dict]:
    """Тестовый набор: n seed-треков разных артистов/жанров из eval-пула."""
    pool = json.loads(POOL.read_text())
    seen_artists: set[str] = set()
    out: list[dict] = []
    for row in pool:
        for t in row.get("top", []):
            tid = t.get("id") or ""
            if not tid.startswith("ym_"):
                continue
            artist = (t.get("artist") or "").strip().lower()
            if not artist or artist in seen_artists:
                continue
            seen_artists.add(artist)
            out.append({"id": tid, "artist": t.get("artist", ""),
                        "title": t.get("title", ""), "query": row.get("query", "")})
            break
        if len(out) >= n:
            break
    return out


def radio(seed: dict, limit: int, timeout: float = 120.0) -> tuple[dict | None, float, str | None]:
    ctx = urllib.parse.quote(f"{seed['artist']} - {seed['title']}")
    url = (f"{BASE}/radio?id={urllib.parse.quote(seed['id'])}&limit={limit}"
           f"&ctx={ctx}&queue=&fresh=1&mode=similar")
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.loads(r.read())
        return d, time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, str(e)


def clap_metrics(results: list[dict], seed: dict) -> tuple[float | None, float]:
    """Средний косинус к seed + покрытие (доля треков с эмбеддингом)."""
    sys.path.insert(0, str(ROOT))
    from app import embeddings as E  # noqa: PLC0415
    E.init_db()
    ref = E.embedding_for(seed["id"], compute=False)
    if ref is None:
        return None, 0.0
    ids = [t["id"] for t in results if t.get("id")]
    sims = E.audio_similarities(ids, ref)
    cov = len(sims) / len(ids) if ids else 0.0
    mean = statistics.mean(sims.values()) if sims else None
    return mean, cov


def covers_count(results: list[dict]) -> int:
    n = 0
    for t in results:
        tl = (t.get("title") or "").lower()
        if any(w in tl for w in ("cover", "кавер", "караоке", "remix", "remix)", "(live",
                                 "трибьют", "tribute", "instrumental")):
            n += 1
    return n


def uniq_at(results: list[dict], k: int) -> int:
    return len({(t.get("artist") or "").strip().lower() for t in results[:k]})


def run(mode: str, n: int, limit: int) -> dict:
    rows = []
    for s in seeds(n):
        d, dt, err = radio(s, limit)
        if err or d is None:
            rows.append({"seed": s, "error": err, "time_s": round(dt, 2)})
            continue
        res = d.get("results", [])
        arts = [(t.get("artist") or "").strip().lower() for t in res]
        seed_a = s["artist"].strip().lower()
        share = (sum(1 for a in arts if a and seed_a in a) / len(res)) if res else 0.0
        try:
            clap_mean, clap_cov = clap_metrics(res, s)
        except Exception:  # noqa: BLE001
            clap_mean, clap_cov = None, 0.0
        rows.append({
            "seed": s,
            "n": len(res),
            "uniq5": uniq_at(res, 5), "uniq10": uniq_at(res, 10), "uniq20": uniq_at(res, 20),
            "share_seed": round(share, 3),
            "dup": len(res) - len({t.get("id") for t in res}),
            "covers": covers_count(res),
            "clap_mean": round(clap_mean, 4) if clap_mean is not None else None,
            "clap_cov": round(clap_cov, 3),
            "time_s": round(dt, 2),
            "say": (d.get("say") or "")[:60],
        })
        print(f"  {s['artist'][:22]:24} n={len(res):2} uniq20={rows[-1]['uniq20']:2} "
              f"share={share:.2f} clap={rows[-1]['clap_mean']} {dt:.1f}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    def agg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None
    summary = {
        "mode": mode, "seeds": len(rows), "ok": len(ok),
        "errors": len(rows) - len(ok),
        "n_mean": agg("n"), "n_min": min((r["n"] for r in ok), default=0),
        "uniq20_mean": agg("uniq20"), "uniq10_mean": agg("uniq10"), "uniq5_mean": agg("uniq5"),
        "share_seed_mean": agg("share_seed"),
        "dup_total": sum(r.get("dup", 0) for r in ok),
        "covers_total": sum(r.get("covers", 0) for r in ok),
        "clap_mean": agg("clap_mean"), "clap_cov_mean": agg("clap_cov"),
        "time_mean_s": agg("time_s"), "time_max_s": max((r["time_s"] for r in ok), default=0),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=40, help="сколько seed-треков")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"),
                    help="сравнить два JSON-результата")
    args = ap.parse_args()

    if args.compare:
        a = json.loads((ROOT / "research" / f"similar_bench_{args.compare[0]}.json").read_text())
        b = json.loads((ROOT / "research" / f"similar_bench_{args.compare[1]}.json").read_text())
        sa, sb = a["summary"], b["summary"]
        print(f"{'метрика':22} {'old':>10} {'new':>10} {'дельта':>10}")
        for k in sa:
            if k in ("mode", "seeds", "ok", "errors"):
                continue
            va, vb = sa[k], sb[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = round(vb - va, 3)
                print(f"{k:22} {va:>10} {vb:>10} {d:>+10}")
            else:
                print(f"{k:22} {va!s:>10} {vb!s:>10}")
        return

    out = Path(args.out or ROOT / f"research/similar_bench_{args.mode}.json")
    print(f"=== similar bench [{args.mode}] seeds={args.n} limit={args.limit} ===")
    data = run(args.mode, args.n, args.limit)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("\nСВОДКА:", json.dumps(data["summary"], ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
