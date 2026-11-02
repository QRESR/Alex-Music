#!/usr/bin/env python3
"""Integration API prototype (exactly what will ship):

    get_embedding(track_bytes) -> list[float]      # 512-dim, L2-normalised
    get_text_embedding(text)   -> list[float]      # same space

Design: singleton model, fixed 10s windows, mean-pool + L2 (deterministic),
batches of windows, serialised by a lock so only one forward pass at a time.
"""
import inspect
import io
import threading
import time

import numpy as np
import soundfile as sf
import soxr
import torch

MODEL_ID = "laion/clap-htsat-unfused"   # Apache-2.0, verified working
SR = 48000
WINDOW_S = 10.0     # == feature extractor max_length_s -> no random truncation
BATCH = 6           # windows per forward pass
MIN_SEG_S = 1.0

_lock = threading.Lock()
_model = _proc = _audio_kw = None


def set_threads(n: int) -> None:
    torch.set_num_threads(n)


def _load():
    global _model, _proc, _audio_kw
    if _model is None:
        from transformers import ClapModel, ClapProcessor

        _proc = ClapProcessor.from_pretrained(MODEL_ID)
        _model = ClapModel.from_pretrained(MODEL_ID).eval()
        _audio_kw = "audio" if "audio" in inspect.signature(_proc.__call__).parameters else "audios"
    return _model, _proc


def _decode(track_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(track_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    return wav


def _norm(out) -> torch.Tensor:
    if not torch.is_tensor(out):          # transformers>=5 returns a model output
        out = out.pooler_output
    return torch.nn.functional.normalize(out.float(), dim=-1)


def get_embedding(track_bytes: bytes) -> list[float]:
    with _lock:
        model, proc = _load()
        wav = _decode(track_bytes)
        win = int(WINDOW_S * SR)
        segs = [wav[i: i + win] for i in range(0, len(wav), win)]
        segs = [s for s in segs if len(s) >= MIN_SEG_S * SR]
        if not segs:
            segs = [wav]
        parts = []
        for i in range(0, len(segs), BATCH):
            inputs = proc(**{_audio_kw: segs[i: i + BATCH]}, sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                parts.append(_norm(model.get_audio_features(**inputs)))
        vec = torch.nn.functional.normalize(torch.cat(parts).mean(dim=0), dim=-1)
        return vec.squeeze(0).tolist()


def get_text_embedding(text: str) -> list[float]:
    with _lock:
        model, proc = _load()
        inputs = proc(text=[text], return_tensors="pt")
        with torch.inference_mode():
            vec = _norm(model.get_text_features(**inputs))
        return vec.squeeze(0).tolist()


if __name__ == "__main__":
    import sys

    set_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
    A = "/root/.hermes/profiles/gera/cache/scratch/audio-emb/"
    files = ["MJ_PIA_Solarity_AcidTechno.ogg", "Matt_s_Blues.mp3"]
    for f in files:
        data = open(A + f, "rb").read()
        t0 = time.perf_counter()
        emb = get_embedding(data)
        dt = time.perf_counter() - t0
        print(f"{f:35s} dim={len(emb)} norm={np.linalg.norm(emb):.4f} time={dt:.2f}s (incl. first-call model load)")
    # determinism + agreement with the offline chunked pipeline
    import numpy as np

    e1 = np.array(get_embedding(open(A + files[0], "rb").read()))
    e2 = np.array(get_embedding(open(A + files[0], "rb").read()))
    ref = np.load(A + "emb/techno_ck_unfused.npy")
    print(f"repeat cos={float(e1 @ e2):.6f}   vs offline pipeline cos={float(e1 @ ref):.6f}")
    t = get_text_embedding("acid techno with four on the floor beat")
    print("text dim:", len(t), "cos(text, techno) =", round(float(np.array(t) @ e1), 3))
