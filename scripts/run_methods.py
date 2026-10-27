"""Run a list of decoding methods with one fine-tuned checkpoint and write
quality + cost metrics to a JSON file (one entry per method).

    python -m scripts.run_methods --ckpt checkpoints/T1_varlen/best.pt \
        --split dev --methods direct_cut serial_yu serial_tang serial_teacher \
        serial_cross parallel --out results/E4_dev.json

Predictions are also saved (npz) so later scripts (bootstrap, recompute)
read them instead of re-running the model.
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
from punct.runner import dump, load_split, run_method, score  # noqa: E402

PARALLEL_DEFAULT = dict(window=400, overlap=100, tail="short", mask_left=0, mask_right=0, batch_size=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--methods", nargs="+", default=["parallel"])
    ap.add_argument("--limit", type=int, default=0, help="use only the first N articles (smoke runs)")
    ap.add_argument("--window", type=int, default=PARALLEL_DEFAULT["window"])
    ap.add_argument("--overlap", type=int, default=PARALLEL_DEFAULT["overlap"])
    ap.add_argument("--tail", default=PARALLEL_DEFAULT["tail"], choices=["short", "aligned"])
    ap.add_argument("--mask_left", type=int, default=0)
    ap.add_argument("--mask_right", type=int, default=0)
    ap.add_argument("--profile", default="hard", choices=["hard", "gauss"])
    ap.add_argument("--batch_size", type=int, default=PARALLEL_DEFAULT["batch_size"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_preds", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[: args.limit], gold[: args.limit]
    model = load_finetuned(args.bert_dir, args.ckpt, device)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    run_args = dict(window=args.window, overlap=args.overlap, tail=args.tail, mask_left=args.mask_left,
                    mask_right=args.mask_right, profile=args.profile, batch_size=args.batch_size)
    result = {"ckpt": args.ckpt, "split": args.split, "n_articles": len(chars), "run_args": run_args,
              "methods": {}}
    preds_out = {}
    # Per-method checkpointing: after every method the partial result is written, so a crash in
    # a later method does not discard earlier ones. A restart with the same settings skips them.
    part_json, part_npz = args.out + ".partial.json", args.out + ".partial.npz"
    if os.path.exists(part_json) and os.path.exists(part_npz):
        with open(part_json, encoding="utf-8") as fh:
            prev = json.load(fh)
        same = all(prev.get(k) == result[k] for k in ("ckpt", "split", "n_articles", "run_args"))
        if same:
            with np.load(part_npz) as z:
                preds_out = {k: z[k] for k in z.files}
            result["methods"] = {k: v for k, v in prev["methods"].items() if k in preds_out}
            print(f"resuming: already done {list(result['methods'])}", flush=True)
        else:
            print("partial result found but settings differ; ignoring it", flush=True)
    for name in args.methods:
        if name in result["methods"]:
            print(f"{name}: skipped (done in an earlier attempt)", flush=True)
            continue
        kw = {}
        if name == "parallel":
            kw = dict(window=args.window, overlap=args.overlap, tail=args.tail, mask_left=args.mask_left,
                      mask_right=args.mask_right, profile=args.profile, batch_size=args.batch_size)
        elif name == "tang_parallel":
            kw = dict(window=125, overlap=20, tail="aligned", batch_size=args.batch_size)
        elif name == "direct_cut":
            kw = dict(window=64, batch_size=args.batch_size)
        pred, stats = run_method(name, chars, logits_fn, special, device, **kw)
        entry = {"cost": stats, "quality": score(gold, pred)}
        if name == "parallel":
            entry["config"] = kw
        result["methods"][name] = entry
        preds_out[name] = np.concatenate(pred)
        print(f"{name}: micro_f1={entry['quality']['micro_f1']:.4f} "
              f"chars/s={stats['chars_per_second']:.0f} seconds={stats['seconds']:.1f}", flush=True)
        # save after every method so a later crash keeps this one
        np.savez_compressed(part_npz, **preds_out)
        dump(part_json, result)
    dump(args.out, result)
    if args.save_preds:
        np.savez_compressed(args.save_preds, **preds_out)
    for p in (part_json, part_npz):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
