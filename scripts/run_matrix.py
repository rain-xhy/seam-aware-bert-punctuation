"""E2 / E1 / E8: models x inference windows matrix.

    python -m scripts.run_matrix --split dev \
        --models T1_varlen=checkpoints/T1_varlen/best.pt T2_fixed256=checkpoints/T2_fixed256/best.pt \
        --windows 64 125 400 510 --out results/E2_matrix_dev.json

Every model is decoded with the same parallel engine on the same articles at
each window size; overlap is window // 4 unless --overlaps gives one value per
window. A model whose checkpoint is missing is recorded as skipped (not
silently dropped), so a partial matrix is visible as partial. The same script
serves E1 (class weighting), E8 (edge-aware on/off) and the centre-only
control: pass those checkpoints as extra --models entries.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import dump, load_split, run_method, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint.pt pairs")
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--windows", type=int, nargs="+", default=[64, 125, 400, 510])
    ap.add_argument("--overlaps", type=int, nargs="+", default=[],
                    help="one overlap per window; default window // 4")
    ap.add_argument("--tail", default="short", choices=["short", "aligned"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.overlaps and len(args.overlaps) != len(args.windows):
        ap.error("--overlaps needs one value per --windows entry")
    overlaps = args.overlaps or [w // 4 for w in args.windows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[: args.limit], gold[: args.limit]

    result = {"split": args.split, "n_articles": len(chars), "tail": args.tail,
              "windows": args.windows, "overlaps": overlaps, "models": {}}
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"bad --models entry {spec!r}, expected name=path")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            result["models"][name] = {"skipped": f"missing checkpoint {path}"}
            print(f"{name}: SKIPPED (no {path})", flush=True)
            dump(args.out, result)
            continue
        fn = make_logits_fn(load_finetuned(args.bert_dir, path, device))
        cells = {}
        for w, v in zip(args.windows, overlaps):
            pred, stats = run_method("parallel", chars, fn, special, device, window=w, overlap=v,
                                     tail=args.tail, batch_size=args.batch_size)
            q = score(gold, pred)
            cells[str(w)] = {"window": w, "overlap": v, "ckpt": path,
                             "micro_f1": q["micro_f1"], "macro_f1": q["macro_f1"],
                             "seam_fixed": q["seam_fixed"], "ter": stats["ter"]}
            print(f"{name} W={w} V={v}: F1={q['micro_f1']:.4f} TER={stats['ter']:.3f}", flush=True)
            result["models"][name] = {"cells": cells}
            dump(args.out, result)
        del fn
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
