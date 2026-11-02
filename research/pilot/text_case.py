#!/usr/bin/env python3
"""Text -> music ranking pilot: text embeddings vs cached audio embeddings."""
import argparse, json, time

import numpy as np
import torch

TRACKS = ["chopin (classical piano)", "blues (electric blues)", "techno (acid techno)", "rock (psychedelic rock)"]
FILES = ["chopin_ck.npy", "blues_ck.npy", "techno_ck.npy", "rock_ck.npy"]
FILES_UNFUSED = ["chopin_ck_unfused.npy", "blues_ck_unfused.npy", "techno_ck_unfused.npy", "rock_ck_unfused.npy"]
PROMPTS = [
    "classical solo piano piece",
    "slow blues with electric guitar",
    "psychedelic rock guitar jam",
    "acid techno with four on the floor beat",
    "cheerful pop song with female vocals",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--unfused", action="store_true")
    args = ap.parse_args()

    from transformers import ClapModel, ClapProcessor

    t0 = time.perf_counter()
    model = ClapModel.from_pretrained(args.model).eval()
    proc = ClapProcessor.from_pretrained(args.model)
    t_load = time.perf_counter() - t0

    fe = getattr(proc, "feature_extractor", proc)
    kw = "text"

    t0 = time.perf_counter()
    inputs = proc(**{kw: PROMPTS}, return_tensors="pt", padding=True)
    with torch.inference_mode():
        f = model.get_text_features(**inputs)
    if not torch.is_tensor(f):
        f = f.pooler_output
    f = torch.nn.functional.normalize(f.float(), dim=-1)
    t_text = time.perf_counter() - t0

    with open("/proc/self/status") as fh:
        rss = next(int(l.split()[1]) / 1024 for l in fh if l.startswith("VmHWM:"))

    files = FILES_UNFUSED if args.unfused else FILES
    idx = [i for i, x in enumerate(files) if x]
    embs = np.stack([np.load(f"{args.dir}/{files[i]}") for i in idx])
    names = [TRACKS[i] for i in idx]

    sim = (f.numpy() @ embs.T)  # (prompts, tracks)
    out = {"model": args.model, "load_s": round(t_load, 2), "text_infer_s": round(t_text, 3),
           "peak_rss_mb": round(rss, 0), "ranking": []}
    print(json.dumps(out["model"]), "load", out["load_s"], "text_s", out["text_infer_s"], "rss", out["peak_rss_mb"])
    for pi, p in enumerate(PROMPTS):
        order = np.argsort(-sim[pi])
        ranks = [(names[j], round(float(sim[pi][j]), 3)) for j in order]
        print(f"  '{p}'")
        for n, s in ranks:
            print(f"      {s:+.3f}  {n}")
        out["ranking"].append({"prompt": p, "order": [names[j] for j in order],
                               "scores": [round(float(sim[pi][j]), 4) for j in order]})
    print("JSON:" + json.dumps(out))


if __name__ == "__main__":
    main()
