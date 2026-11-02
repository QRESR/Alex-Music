#!/usr/bin/env python3
"""Handcrafted-feature fallback benchmark (librosa on py3.14): timing, RAM, similarity matrix."""
import json, os, resource, time

import numpy as np
import librosa

FILES = {
    "chopin": "Chopin_Mazurka_in_A_flat_major_B_85.flac",
    "blues": "Matt_s_Blues.mp3",
    "techno": "MJ_PIA_Solarity_AcidTechno.ogg",
    "rock": "MokkaMusic_Exotic_Jam.ogg",
}
D = "/root/.hermes/profiles/gera/cache/scratch/audio-emb"
SR = 22050


def feats(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    t0 = time.perf_counter()
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    poly = librosa.feature.spectral_flatness(y=y)
    t_feat = time.perf_counter() - t0
    v = np.concatenate([
        mf.mean(axis=1), mf.std(axis=1),
        ch.mean(axis=1),
        [sc.mean(), rms.mean(), rms.std(), poly.mean()],
    ])
    v = v / (np.linalg.norm(v) + 1e-9)
    return v, t_feat, len(y) / sr


# warm-up (numba JIT compilation of stft/rhythm paths)
feats(os.path.join(D, FILES["blues"]))

res = {}
E, times = {}, {}
for name, f in FILES.items():
    v, t, dur = feats(os.path.join(D, f))
    E[name], times[name] = v, t
    print(f"{name:8s} track={dur:6.1f}s  feature_time={t:.2f}s  dim={v.shape[0]}")

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
names = list(FILES)
print("\ncosine matrix (handcrafted):")
for i, n in enumerate(names):
    print("  ", n.ljust(8), "  ".join(f"{float(E[n] @ E[m]):+.3f}" for m in names))

avg_time = float(np.mean(list(times.values())))
print(f"\navg feature time/track: {avg_time:.2f}s   peak RSS: {peak:.0f} MB")
print("JSON:" + json.dumps({"avg_feature_s": round(avg_time, 2), "peak_rss_mb": round(peak, 0),
                            "dims": int(v.shape[0])}))
