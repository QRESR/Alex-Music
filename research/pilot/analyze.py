#!/usr/bin/env python3
"""Similarity sanity checks over saved embeddings (cosine matrices, determinism)."""
import itertools, json, os, sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/root/.hermes/profiles/gera/cache/scratch/audio-emb/emb"


def load(name):
    return np.load(os.path.join(D, name))


def cos(a, b):
    return float(a @ b)


def matrix(names, title):
    print(f"\n== cosine matrix: {title}")
    E = np.stack([load(n) for n in names])
    header = " " * 12 + "  ".join(f"{n[:8]:>8s}" for n in names)
    print(header)
    for i, n in enumerate(names):
        row = "  ".join(f"{float(E[i] @ E[j]):8.3f}" for j in range(len(names)))
        print(f"{n[:11]:>12s}  {row}")
    off = [float(E[i] @ E[j]) for i, j in itertools.combinations(range(len(names)), 2)]
    print(f"  off-diagonal: min={min(off):.3f} max={max(off):.3f} mean={np.mean(off):.3f}")


res = {}
matrix(["chopin_ck.npy", "blues_ck.npy", "techno_ck.npy", "rock_ck.npy"], "larger_clap_music, chunked (4 tracks)")
res["full_run_cosine"] = round(cos(load("techno_full1.npy"), load("techno_full2.npy")), 4)
res["chunked_rerun_cosine"] = round(cos(load("techno_ck.npy"), load("techno_ck2.npy")), 4)
res["full_vs_chunked_cosine"] = round(cos(load("techno_full1.npy"), load("techno_ck.npy")), 4)
res["excerpt_vs_chunked_cosine"] = round(cos(np.load("/tmp/emb_techno_30.npy"), load("techno_ck.npy")), 4)
res["chunked_vs_batch6_cosine"] = round(cos(load("rock_ck.npy"), load("rock_ckb6.npy")), 4)

print("\n== determinism / truncation effects")
print("  full run1 vs full run2 (same file):   cos =", res["full_run_cosine"], "-> random 10s crop")
print("  chunked run1 vs chunked run2:         cos =", res["chunked_rerun_cosine"], "-> deterministic")
print("  full(crop) vs chunked(full track):    cos =", res["full_vs_chunked_cosine"])
print("  excerpt30 vs chunked:                 cos =", res["excerpt_vs_chunked_cosine"])
print("  chunked batch=1 vs batch=6:           cos =", res["chunked_vs_batch6_cosine"])

print("\n== general CLAP (clap-htsat-unfused) vs music CLAP on same tracks")
for a, b, label in [("techno_ck.npy", "techno_ck_unfused.npy", "techno"),
                    ("blues_ck.npy", "blues_ck_unfused.npy", "blues")]:
    print(f"  {label}: cos(music_clap, general_clap) = {round(cos(load(a), load(b)), 3)}")

matrix(["chopin_ck_unfused.npy", "blues_ck_unfused.npy", "techno_ck_unfused.npy", "rock_ck_unfused.npy"],
       "clap-htsat-unfused, chunked (4 tracks)")
matrix(["chopin_ck_mms.npy", "blues_ck_mms.npy", "techno_ck_mms.npy", "rock_ck_mms.npy"],
       "larger_clap_music_and_speech, chunked (4 tracks)")

print("\nJSON:" + json.dumps(res))
