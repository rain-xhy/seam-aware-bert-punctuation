"""Agreement of parallel decoding labels across batch sizes and across precisions.

Runs the default parallel configuration (W=400, V=100, short tail) on the first N
articles of a split with bf16 autocast (the setting used everywhere else) and with
plain fp32, and compares the label sequences position by position. A batch-size
comparison under bf16 is included as well.

    python -m scripts.verify_precision --ckpt checkpoints/T1_varlen/best.pt \
        --split test --limit 300 --out results/precision_consistency.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import load_split, run_method  # noqa: E402


def agreement(a, b):
    fa, fb = np.concatenate(a), np.concatenate(b)
    if fa.shape != fb.shape:
        return {"consistent": False, "reason": "length mismatch"}
    same = int((fa == fb).sum())
    return {"consistent": same == fa.size, "matches": same, "total": int(fa.size),
            "mismatched_positions": int(fa.size - same), "agreement_rate": same / max(fa.size, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    chars, _ = load_split(args.data_dir, args.split)
    chars = chars[:args.limit]
    model = load_finetuned(args.bert_dir, args.ckpt, device)
    fn_bf16 = make_logits_fn(model, use_bf16=True)
    fn_fp32 = make_logits_fn(model, use_bf16=False)

    def run(fn, bs):
        pred, st = run_method("parallel", chars, fn, special, device, window=args.window,
                              overlap=args.overlap, tail="short", batch_size=bs)
        print(f"bs={bs} seconds={st['seconds']:.1f}", flush=True)
        return pred

    preds = {}
    for bs in args.batches:
        preds[("bf16", bs)] = run(fn_bf16, bs)
    ref_bs = 4 if 4 in args.batches else args.batches[0]
    preds[("fp32", ref_bs)] = run(fn_fp32, ref_bs)

    out = {"ckpt": args.ckpt, "split": args.split, "n_articles": len(chars),
           "window": args.window, "overlap": args.overlap, "comparisons": []}
    ref = preds[("bf16", args.batches[0])]
    for bs in args.batches[1:]:
        c = agreement(ref, preds[("bf16", bs)])
        c.update(a=f"bf16 batch={args.batches[0]}", b=f"bf16 batch={bs}")
        out["comparisons"].append(c)
    c = agreement(preds[("bf16", ref_bs)], preds[("fp32", ref_bs)])
    c.update(a=f"bf16 batch={ref_bs}", b=f"fp32 batch={ref_bs}")
    out["comparisons"].append(c)
    for c in out["comparisons"]:
        print(c, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
