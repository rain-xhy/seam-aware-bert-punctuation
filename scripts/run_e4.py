"""E4: efficiency comparison with the interleaved timing protocol, plus the
consistency checks (batch size, bf16 vs fp32) on the same texts.

    python -m scripts.run_e4 --ckpt checkpoints/T1_varlen/best.pt --split dev \
        --limit 300 --repeats 5 --out results/E4_dev.json

Timed configurations (all use the same fine-tuned model):
    serial_teacher   400/300 serial, one article at a time
    serial_cross     400/300 serial, cross-article batches
    parallel_b1      parallel windows, batch size 1 (same windows as parallel)
    parallel         parallel windows, --batch_size
Quality of the last run of each configuration is scored with punct.runner.score.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.bench import agreement, time_interleaved  # noqa: E402
from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import dump, load_split, run_method, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--tail", default="short", choices=["short", "aligned"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[: args.limit], gold[: args.limit]
    model = load_finetuned(args.bert_dir, args.ckpt, device)
    fn_bf16 = make_logits_fn(model, use_bf16=True)
    fn_fp32 = make_logits_fn(model, use_bf16=False)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    pkw = dict(window=args.window, overlap=args.overlap, tail=args.tail)

    def job(name, fn, **kw):
        return lambda: run_method(name, chars, fn, special, device, **kw)

    configs = {
        "serial_teacher": job("serial_teacher", fn_bf16),
        "serial_cross": job("serial_cross", fn_bf16),
        "parallel_b1": job("parallel", fn_bf16, batch_size=1, **pkw),
        "parallel": job("parallel", fn_bf16, batch_size=args.batch_size, **pkw),
    }
    timing = time_interleaved(configs, repeats=args.repeats, warmup=1)

    total = int(sum(len(c) for c in chars))
    result = {"ckpt": args.ckpt, "split": args.split, "n_articles": len(chars), "text_chars": total,
              "parallel_config": pkw, "batch_size": args.batch_size, "repeats": args.repeats, "methods": {}}
    for name, t in timing.items():
        pred, stats = t["last_result"]
        result["methods"][name] = {
            "wall_median_s": t["median"], "wall_std_s": t["std"], "wall_min_s": t["min"],
            "wall_max_s": t["max"], "runs_s": t["runs"],
            "chars_per_second_median": total / t["median"],
            "cost_last_run": stats, "quality": score(gold, pred),
        }
        print(f"{name}: median={t['median']:.2f}s std={t['std']:.2f}s "
              f"F1={result['methods'][name]['quality']['micro_f1']:.4f}", flush=True)

    base = result["methods"]["serial_teacher"]["wall_median_s"]
    b1 = result["methods"]["parallel_b1"]["wall_median_s"]
    for name, m in result["methods"].items():
        m["speedup_vs_serial_teacher"] = base / m["wall_median_s"]
        m["speedup_vs_parallel_b1"] = b1 / m["wall_median_s"]

    # consistency: batch size and precision (same windows, same texts)
    p_b1 = timing["parallel_b1"]["last_result"][0]
    p_bs = timing["parallel"]["last_result"][0]
    p_fp32, _ = run_method("parallel", chars, fn_fp32, special, device, batch_size=args.batch_size, **pkw)
    result["consistency"] = {
        "batch1_vs_batchN_bf16": agreement(p_b1, p_bs),
        "bf16_vs_fp32": agreement(p_bs, p_fp32),
    }
    print("consistency:", result["consistency"], flush=True)
    dump(args.out, result)


if __name__ == "__main__":
    main()
