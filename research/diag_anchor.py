#!/usr/bin/env python3
"""Диагностика якоря на проблемных запросах + проверка кандидатов."""
import sys
sys.path.insert(0, '.')

from app import recommender as rec

CASES = [
    "Зомби remastered",
    "песни из очень странные дела",
    "зомберы мастерят",
    "асти бриллиант",
    "баста без слим шейди",
    "The Cranberries",
]

print("=== _extract_artist_from_query + query_artist_candidates ===")
for q in CASES:
    ext = rec._extract_artist_from_query(q)
    cands = rec.query_artist_candidates(q)
    print(f"{q!r}")
    print(f"   _extract -> {ext!r}")
    print(f"   cands    -> {cands}")

print("\n=== проверка: remastered/дела/мастерят проходят ли фильтры ===")
for w in ["remastered", "дела", "мастерят", "баста", "зомби", "струнный"]:
    lw = w.lower().replace("ё", "е")
    in_non = lw in rec._NON_ARTIST_WORDS
    suffix_hit = len(lw) >= 5 and lw.endswith(rec._SUFFIXES)
    substr = any(g in lw for g in (
        "поп", "рэп", "рок", "джаз", "музык", "песн", "хит", "ремикс",
        "подбор", "танц", "спок", "весел", "весёл", "груст", "бодр",
        "электр", "зарубеж", "русск", "современ", "для", "тренир",
        "включи", "постав", "поигра", "запусти", "найди"))
    verdict = "КАНДИДАТ-АРТИСТ (пропускается!)" if not (in_non or suffix_hit or substr) else "отсеян"
    print(f"  {w!r}: non={in_non} suffix={suffix_hit} substr={substr} -> {verdict}")
