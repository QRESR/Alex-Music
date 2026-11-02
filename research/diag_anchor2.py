#!/usr/bin/env python3
"""Диагностика: как anchor-логика обрабатывает проблемные запросы на РЕАЛЬНОМ пуле.

Цель: показать, какой seed выбирается и почему обрезается (без запуска сервера).
"""
import sys, asyncio
sys.path.insert(0, '.')

from app import recommender as rec


class T:
    def __init__(self, artist, title):
        self.artist = artist
        self.title = title
        self.id = f"{artist}|{title}"


async def ym(q, n=15):
    from app import ym_client
    return await ym_client.search(q, limit=n)


def simulate(q, tracks):
    """Воспроизводит логику выбора seed + anchor-фильтра из main.py."""
    print(f"\n{'='*70}\nЗАПРОС: {q!r}  (пул {len(tracks)})")
    # 1. выбор seed как в main
    ca_seed, ca_hits = rec.best_seed_from_query(q, tracks)
    seed_anchor = (
        ca_seed
        or rec._extract_artist_from_query(q)
    )
    print(f"  best_seed -> {ca_seed!r} (hits={ca_hits})")
    print(f"  _extract  -> {rec._extract_artist_from_query(q)!r}")
    print(f"  => SEED_ANCHOR = {seed_anchor!r}")
    if not (seed_anchor and len(seed_anchor) >= 3):
        print("  => якорь НЕ применяется (нет seed) — выдача = весь пул")
        return
    a_toks = set(rec._norm(seed_anchor).split())

    def match_artist(t):
        art = rec._norm(t.artist or "")
        return any(tok in art for tok in a_toks if len(tok) >= 3)

    def match_title(t):
        ttl = rec._norm(t.title or "")
        return any(tok in ttl for tok in a_toks if len(tok) >= 3)

    artist_hits = [t for t in tracks if match_artist(t)]
    print(f"  artist_hits = {len(artist_hits)}")
    # strong
    SVC = {"the","a","an","and","of","feat","ft","with","x","vs","or","in","on","deb","la","le","di","de"}
    def strong(a):
        arts=[w for w in rec._norm(a).split() if w.isalnum()]
        extra=[w for w in arts if w not in a_toks and w not in SVC]
        sh = bool(a_toks) and (a_toks & set(arts))
        return len(extra)<=1 or (sh and len(extra)<=3)
    strong_hits=[t for t in artist_hits if strong(t.artist or "")]
    print(f"  strong_hits = {len(strong_hits)}")

    if len(artist_hits) >= 3 or strong_hits:
        result = artist_hits
        branch = "СТРОГИЙ (artist_hits>=3 or strong)"
    else:
        result = [t for t in tracks if match_artist(t) or match_title(t)]
        branch = "title-матч fallback"
    print(f"  ветка: {branch} -> {len(result)} треков")
    if not result:
        print(f"  => ПУСТО -> возврат _raw_anchor (все {len(tracks)})")
        return
    for i, t in enumerate(result[:6]):
        print(f"    {i+1}. {t.artist} - {t.title[:50]}")


async def main():
    cases = [
        ("Зомби remastered", "yandex"),
        ("песни из очень странные дела", "yandex"),
        ("асти бриллиант", "yandex"),
    ]
    for q, _ in cases:
        raw = await ym(q)
        tracks = [T(d.get("artist",""), d.get("title","")) for d in raw]
        simulate(q, tracks)

asyncio.run(main())
