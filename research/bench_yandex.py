#!/usr/bin/env python3
"""Бенч Яндекс-поиска: 100 запросов из bench_100_queries.json -> TSV отчёта.

Метрики по каждому запросу: n, время, топ-5 треков (артист - название),
авто-оценки:
  - empty: ни одного трека
  - slow: > 15с
  - artist_hit: для cat=artist — есть ли искомый артист в топ-5 (через
    token overlap, толпастые совпадения с учётом транслита не нужны —
    простая проверка подстроки обоих имён)
  - anchor_word: какое слово стало якорем (из журнала отдельно)
Записывает research/bench_yandex_results.tsv + полный JSON.
"""
import json, sys, time, urllib.parse, urllib.request, pathlib, re

BASE = "http://127.0.0.1:8011"
HERE = pathlib.Path(__file__).parent
QUERIES = json.loads((HERE / "bench_100_queries.json").read_text())

def search(q: str, source: str = "yandex", limit: int = 10, timeout: int = 60):
    url = f"{BASE}/search?q={urllib.parse.quote(q)}&limit={limit}&mix=0&source={source}"
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return time.time() - t0, data, None
    except Exception as e:
        return time.time() - t0, None, str(e)

def artist_in_top(q: str, results: list, k: int = 5) -> bool:
    """Есть ли все слова артиста (q) среди топ-k артистов (без учёта регистра)."""
    toks = [t for t in re.split(r"\s+", q.lower()) if t]
    blob = " | ".join((t.get("artist") or "").lower() for t in results[:k])
    return all(t in blob for t in toks)

rows = []
for i, item in enumerate(QUERIES, 1):
    q, cat = item["q"], item["cat"]
    dt, data, err = search(q)
    if err or data is None:
        rows.append({"i": i, "cat": cat, "q": q, "n": 0, "dt": round(dt, 1),
                     "err": err or "unknown", "top": [], "artist_hit": None})
        print(f"[{i:3d}/{len(QUERIES)}] {cat:7s} {q[:44]!r} ERR {err}", flush=True)
        continue
    res = data.get("results") or []
    top = [f"{t.get('artist','')} - {t.get('title','')}" for t in res[:5]]
    hit = artist_in_top(q, res) if cat == "artist" else None
    rows.append({"i": i, "cat": cat, "q": q, "n": len(res), "dt": round(dt, 1),
                 "intent": data.get("intent"), "say": (data.get("say") or "")[:70],
                 "top": top, "artist_hit": hit, "err": None})
    flag = "!!" if (len(res) == 0 or dt > 15) else "  "
    print(f"[{i:3d}/{len(QUERIES)}]{flag}{cat:7s} {q[:44]!r} n={len(res)} {dt:.1f}s", flush=True)

(HERE / "bench_yandex_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
with open(HERE / "bench_yandex_results.tsv", "w") as f:
    f.write("i\tcat\tn\tdt\tq\ttop1\ttop2\ttop3\terr\n")
    for r in rows:
        top = r.get("top") or ["", "", ""]
        f.write(f"{r['i']}\t{r['cat']}\t{r['n']}\t{r['dt']}\t{r['q']}\t"
                f"{top[0] if len(top)>0 else ''}\t{top[1] if len(top)>1 else ''}\t"
                f"{top[2] if len(top)>2 else ''}\t{r.get('err') or ''}\n")

n_err = sum(1 for r in rows if r.get("err"))
n_empty = sum(1 for r in rows if not r.get("err") and r["n"] == 0)
n_slow = sum(1 for r in rows if not r.get("err") and r["dt"] > 15)
hits = [r for r in rows if r.get("artist_hit") is not None]
n_hit = sum(1 for r in hits if r["artist_hit"])
print("\n=== SUMMARY ===", flush=True)
print(f"total={len(rows)} err={n_err} empty={n_empty} slow(>15s)={n_slow} "
      f"artist_hit={n_hit}/{len(hits)}", flush=True)
