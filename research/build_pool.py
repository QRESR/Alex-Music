#!/usr/bin/env python3
"""Собрать пул треков (с id!) для CLAP-испытаний из запросов eval'а.

eval JSON хранит top только {artist,title,year} — без id. Для п.8 нужны
track_id → эмбеддинг. Скрипт прходит по тем же запросам /search (серверные
кэши сделают повтор быстрым) и пишет research/eval_pool.json:
    [{query, type, top: [{id, artist, title, year}]}]

Запуск: python3 research/build_pool.py  (сервер должен быть поднят)
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8011"
SRC = Path("/root/.hermes/profiles/gera/cache/scratch/eval/eval_v6.json")
OUT = ROOT / "research" / "eval_pool.json"


def search(q: str, limit: int = 5, mix: int = 1) -> list[dict]:
    url = f"{BASE}/search?q={urllib.parse.quote(q)}&limit={limit}&mix={mix}"
    with urllib.request.urlopen(url, timeout=90) as r:
        d = json.loads(r.read().decode())
    return [
        {"id": t.get("id"), "artist": t.get("artist"), "title": t.get("title"),
         "year": t.get("year")}
        for t in d.get("results", [])
        if (t.get("id") or "").startswith("ym_")
    ]


def main() -> None:
    src = json.loads(SRC.read_text())
    rows, seen_q, uniq = [], set(), {}
    for r in src["results"]:
        q = r.get("query") or ""
        if not q or q in seen_q or "error" in r:
            continue
        seen_q.add(q)
        try:
            top = search(q)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {q[:40]}: {e}", flush=True)
            continue
        for t in top:
            uniq.setdefault(t["id"], t)
        rows.append({"query": q, "type": r.get("type") or "mix", "top": top})
        print(f"  [{len(rows)}] {q[:40]} -> {len(top)}", flush=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"POOL: queries={len(rows)} unique_tracks={len(uniq)} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
