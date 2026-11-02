#!/usr/bin/env python3
"""Probe: are audio embeddings degenerate? noise/sine/silence vs real track, plus text scores."""
import json, sys, time

import numpy as np
import torch

MODEL = sys.argv[1]
SR = 48000
rng = np.random.default_rng(0)

synth = {
    "white_noise": (rng.standard_normal(10 * SR) * 0.1).astype(np.float32),
    "sine_440": (0.2 * np.sin(2 * np.pi * 440 * np.arange(10 * SR) / SR)).astype(np.float32),
    "silence": np.zeros(10 * SR, dtype=np.float32),
    "sine_sweep": (0.2 * np.sin(2 * np.pi * np.cumsum(np.linspace(100, 4000, 10 * SR)) / SR)).astype(np.float32),
}
import soundfile as sf

real, _ = sf.read("/root/.hermes/profiles/gera/cache/scratch/audio-emb/MJ_PIA_Solarity_AcidTechno.ogg",
                  dtype="float32", always_2d=True)
synth["techno_10s"] = real.mean(axis=1)[: 10 * SR]

from transformers import ClapModel, ClapProcessor

t0 = time.perf_counter()
model = ClapModel.from_pretrained(MODEL).eval()
proc = ClapProcessor.from_pretrained(MODEL)
t_load = time.perf_counter() - t0


import inspect

_AUDIO_KW = "audio" if "audio" in inspect.signature(proc.__call__).parameters else "audios"


def _vec(out):
    if not torch.is_tensor(out):
        out = out.pooler_output
    return torch.nn.functional.normalize(out.float(), dim=-1)


def afe(w):
    inputs = proc(**{_AUDIO_KW: w}, sampling_rate=SR, return_tensors="pt")
    f = model.get_audio_features(**inputs)
    return _vec(f).squeeze(0)


def tfe(texts):
    inputs = proc(text=texts, return_tensors="pt", padding=True)
    f = model.get_text_features(**inputs)
    return _vec(f)


names = list(synth)
E = torch.stack([afe(synth[n]) for n in names])
print(f"== {MODEL} (load {t_load:.2f}s)")
print("audio-audio cosines:")
for i, n in enumerate(names):
    print("  ", n.ljust(12), " ".join(f"{float(E[i] @ E[j]):+.3f}" for j in range(len(names))))
# per-dim spread across the 5 inputs
spread = float(E.std(dim=0).mean())
print(f"  mean per-dim std across inputs: {spread:.4f}   ||E||={float(E.norm(dim=-1).mean()):.3f}")

prompts = ["classical solo piano piece", "acid techno with four on the floor beat", "white noise static"]
T = tfe(prompts)
print("text-audio cosines (rows=text, cols=audio):")
for i, p in enumerate(prompts):
    print("  ", p[:34].ljust(35), " ".join(f"{float(T[i] @ E[j]):+.3f}" for j in range(len(names))))
best = {p: names[int(torch.argmax(T[i] @ E.T))] for i, p in enumerate(prompts)}
print("  argmax:", json.dumps(best))
