"""E3: window / overlap / tail-rule sweep with one fine-tuned checkpoint.

    python -m scripts.run_sweep --ckpt checkpoints/T1_varlen/best.pt --split dev \
        --windows 128 256 400 510 --overlaps 0 50 100 200 --tails short aligned \
        --out results/E3_dev.json

Every (window, overlap, tail) triple is decoded with the batched engine and
scored with the full metric set; TER and seconds are recorded so the points
can go straight into the quality-versus-compute plot. With --edep the
EDEP-L/R curves of the first configuration are measured as well.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.edep_scan import edep_scan  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import dump, load_split, run_method, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--windows", type=int, nargs="+", default=[128, 256, 400, 510])
    ap.add_argument("--overlaps", type=int, nargs="+", default=[0, 50, 100, 200])
    ap.add_argument("--tails", nargs="+", default=["short", "aligned"], choices=["short", "aligned"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--edep", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_resume", action="store_true", help="ignore an existing --out file and start over")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[: args.limit], gold[: args.limit]
    logits_fn = make_logits_fn(load_finetuned(args.bert_dir, args.ckpt, device))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    result = {"ckpt": args.ckpt, "split": args.split, "n_articles": len(chars), "points": []}
    if os.path.exists(args.out) and not args.no_resume:
        import json
        with open(args.out, encoding="utf-8") as fh:
            old = json.load(fh)
        if (old.get("ckpt"), old.get("split"), old.get("n_articles")) == (result["ckpt"], result["split"], result["n_articles"]):
            result = old
            print(f"resuming from {args.out}: {len(old['points'])} points done", flush=True)
    done = {(p["config"]["window"], p["config"]["overlap"], p["config"]["tail"]) for p in result["points"]}
    for w in args.windows:
        for v in args.overlaps:
            if v >= w:
                continue
            for tail in args.tails:
                if (w, v, tail) in done:
                    continue
                kw = dict(window=w, overlap=v, tail=tail, batch_size=args.batch_size)
                pred, stats = run_method("parallel", chars, logits_fn, special, device, **kw)
                q = score(gold, pred)
                result["points"].append({"config": kw, "cost": stats, "quality": q})
                print(f"W={w} V={v} tail={tail}: F1={q['micro_f1']:.4f} TER={stats['ter']:.3f} "
                      f"chars/s={stats['chars_per_second']:.0f}", flush=True)
                dump(args.out, result)
    if args.edep and "edep" not in result:
        w = max(args.windows)
        result["edep"] = edep_scan(chars, gold, logits_fn, *special, device, window=w,
                                   overlap=w // 2, batch_size=args.batch_size)
        dump(args.out, result)


if __name__ == "__main__":
    main()
