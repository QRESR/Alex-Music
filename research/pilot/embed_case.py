#!/usr/bin/env python3
"""One inference case per process -> accurate time + peak RSS (VmHWM).

Usage:
  embed_case.py --model M --file F --mode {full,excerpt,chunked}
                [--excerpt 30] [--chunk 10] [--out out.npy] [--threads 4]

Prints a single JSON line to stdout.
"""
import argparse, json, os, sys, time

import numpy as np


def peak_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def decode(path: str, sr: int = 48000):
    import soundfile as sf

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if file_sr != sr:
        import soxr

        wav = soxr.resample(wav, file_sr, sr)
    return wav, sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--mode", default="excerpt", choices=["full", "excerpt", "chunked"])
    ap.add_argument("--excerpt", type=float, default=30.0)
    ap.add_argument("--chunk", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=1, help="windows per forward pass in chunked mode")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.perf_counter()
    wav, sr = decode(args.file)
    t_decode = time.perf_counter() - t0

    import torch

    torch.set_num_threads(args.threads)
    from transformers import ClapModel, ClapProcessor

    t0 = time.perf_counter()
    processor = ClapProcessor.from_pretrained(args.model)
    model = ClapModel.from_pretrained(args.model)
    model.eval()
    t_load = time.perf_counter() - t0

    import inspect

    _audio_kw = "audio" if "audio" in inspect.signature(processor.__call__).parameters else "audios"

    def emb_audio(w):
        inputs = processor(**{_audio_kw: w}, sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            f = model.get_audio_features(**inputs)
        # transformers>=5 may return a model output object instead of a tensor
        if not torch.is_tensor(f):
            if getattr(f, "pooler_output", None) is None:
                raise RuntimeError(f"unexpected audio feature type: {type(f)}")
            f = f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1)

    dur = len(wav) / sr
    t0 = time.perf_counter()
    if args.mode == "excerpt":
        vec = emb_audio(wav[: int(args.excerpt * sr)])
        n = 1
    elif args.mode == "full":
        vec = emb_audio(wav)
        n = 1
    else:  # chunked: mean of per-window embeddings (deterministic, covers whole track)
        win = int(args.chunk * sr)
        segs = []
        for i in range(0, len(wav), win):
            seg = wav[i: i + win]
            if len(seg) < sr:  # drop <1s tail
                break
            segs.append(seg)
        parts = []
        for i in range(0, len(segs), max(1, args.batch)):
            batch = segs[i: i + max(1, args.batch)]
            inputs = processor(**{_audio_kw: batch}, sampling_rate=sr, return_tensors="pt")
            with torch.inference_mode():
                f = model.get_audio_features(**inputs)
            if not torch.is_tensor(f):
                f = f.pooler_output
            parts.append(torch.nn.functional.normalize(f.float(), dim=-1))
        vec = torch.nn.functional.normalize(torch.cat(parts).mean(dim=0), dim=-1)
        n = len(segs)
    t_infer = time.perf_counter() - t0

    out = vec.squeeze(0).numpy().astype(np.float32)
    if args.out:
        np.save(args.out, out)

    print(json.dumps({
        "model": args.model,
        "file": os.path.basename(args.file),
        "mode": args.mode,
        "track_seconds": round(dur, 1),
        "processed_seconds": round(len(wav) / sr if args.mode == "full" else (args.excerpt if args.mode == "excerpt" else args.chunk * n), 1),
        "windows": n,
        "decode_s": round(t_decode, 2),
        "model_load_s": round(t_load, 2),
        "infer_s": round(t_infer, 2),
        "dim": int(out.shape[0]),
        "truncation": getattr(getattr(processor, "feature_extractor", processor), "truncation", "?"),
        "max_length_s": getattr(getattr(processor, "feature_extractor", processor), "max_length_s", "?"),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "threads": args.threads,
    }))


if __name__ == "__main__":
    main()
