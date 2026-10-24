"""E4b: engineering variants and two-pass latency, interleaved timing.

    python -m scripts.run_e4b --ckpt checkpoints/T1_varlen/best.pt --split dev \
        --limit 300 --repeats 5 --out results/E4b_dev.json

Configurations (same model, same texts):
    parallel_plain      decode_parallel, no pinned memory, no prefetch
    parallel_pinned     page-locked host buffers + async copy
    parallel_prefetch   pinned + worker thread encoding the next batch
    tang_parallel       window 125 / overlap 20, drop-last-sentence merge
    two_pass            decode_two_pass (pass 1 finishes, then pass 2)
    two_pass_pipelined  decode_two_pass_pipelined (patch windows share batches)

Run it with the GPU otherwise idle; the timings are meaningless while a
training job is running.
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
from punct.engine import decode_parallel, load_finetuned, make_logits_fn  # noqa: E402
from punct.pipeline import decode_two_pass_pipelined  # noqa: E402
from punct.runner import dump, load_split, run_method, score  # noqa: E402
from punct.two_pass import decode_two_pass  # noqa: E402


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
    ap.add_argument("--conf_thr", type=float, default=0.9)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[: args.limit], gold[: args.limit]
    fn = make_logits_fn(load_finetuned(args.bert_dir, args.ckpt, device))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    pk = dict(window=args.window, overlap=args.overlap, tail=args.tail, batch_size=args.batch_size)
    tk = dict(pk, conf_thr=args.conf_thr)

    def par(**extra):
        def run():
            labels, st = decode_parallel(chars, fn, *special, device, **pk, **extra)
            return labels, {"forward_seconds": st.forward_seconds, "ter": st.ter, "seconds": st.total_seconds}
        return run

    def tang():
        return run_method("tang_parallel", chars, fn, special, device, window=125, overlap=20, tail="aligned",
                          batch_size=args.batch_size)

    def two():
        labels, info = decode_two_pass(chars, fn, special, device, **tk)
        return labels, info

    def piped():
        labels, info = decode_two_pass_pipelined(chars, fn, special, device, **tk)
        return labels, info

    configs = {"parallel_plain": par(), "parallel_pinned": par(pin_memory=True),
               "parallel_prefetch": par(pin_memory=True, prefetch=True), "tang_parallel": tang,
               "two_pass": two, "two_pass_pipelined": piped}
    timing = time_interleaved(configs, repeats=args.repeats, warmup=1)

    total = int(sum(len(c) for c in chars))
    result = {"ckpt": args.ckpt, "split": args.split, "n_articles": len(chars), "text_chars": total,
              "config": tk, "repeats": args.repeats, "methods": {}}
    for name, t in timing.items():
        pred, info = t["last_result"]
        result["methods"][name] = {
            "wall_median_s": t["median"], "wall_std_s": t["std"], "wall_min_s": t["min"],
            "wall_max_s": t["max"], "runs_s": t["runs"], "chars_per_second_median": total / t["median"],
            "cost_last_run": info, "quality": score(gold, pred)}
        print(f"{name}: median={t['median']:.2f}s std={t['std']:.2f}s "
              f"F1={result['methods'][name]['quality']['micro_f1']:.4f}", flush=True)

    plain = result["methods"]["parallel_plain"]["wall_median_s"]
    for n in ("parallel_pinned", "parallel_prefetch"):
        result["methods"][n]["speedup_vs_plain"] = plain / result["methods"][n]["wall_median_s"]
    tp = result["methods"]["two_pass"]["wall_median_s"]
    result["methods"]["two_pass_pipelined"]["speedup_vs_two_pass"] = tp / result["methods"]["two_pass_pipelined"]["wall_median_s"]

    result["consistency"] = {
        "plain_vs_pinned": agreement(timing["parallel_plain"]["last_result"][0], timing["parallel_pinned"]["last_result"][0]),
        "plain_vs_prefetch": agreement(timing["parallel_plain"]["last_result"][0], timing["parallel_prefetch"]["last_result"][0]),
        "two_pass_vs_pipelined": agreement(timing["two_pass"]["last_result"][0], timing["two_pass_pipelined"]["last_result"][0]),
    }
    print("consistency:", result["consistency"], flush=True)
    dump(args.out, result)


if __name__ == "__main__":
    main()
