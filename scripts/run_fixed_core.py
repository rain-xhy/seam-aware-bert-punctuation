"""Fixed-core isolation experiment (完善方案.md 4.2 item 3).

Every article is cut into adjacent cores of `core` characters (the last core
absorbs the remainder, so it is `core`..2*core-1 long). Each core is decoded
inside a window that adds `ctx` characters of context on both sides; only the
labels of the core itself are kept (the context is masked out in the merge).
The seam F1 is measured on the positions within `seam_margin` of the internal
core boundaries, so the only thing that changes between runs is how much
context the model saw around those boundaries.

    python -m scripts.run_fixed_core --ckpt checkpoints/T1_varlen/best.pt \
        --split dev --core 192 --contexts 0 14 31 63 --out results/fixed_core.json
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import PooledWindow, decode_parallel, load_finetuned, make_logits_fn  # noqa: E402
from punct.metrics import punctuation_prf, seam_f1_fixed_positions  # noqa: E402
from punct.runner import dump, load_split  # noqa: E402


def core_boundaries(n: int, core: int) -> List[int]:
    """Internal core boundaries of an article of length n (start offsets of cores 1..k-1)."""
    k = max(n // core, 1)
    return [i * core for i in range(1, k)]


def fixed_core_windows(lengths: Sequence[int], core: int, ctx: int) -> List[PooledWindow]:
    """One window per core: the core plus ctx characters on each side (clipped to the article)."""
    out = []
    for a, n in enumerate(lengths):
        cuts = [0] + core_boundaries(n, core) + [n]
        for s, e in zip(cuts[:-1], cuts[1:]):
            out.append(PooledWindow(a, max(0, s - ctx), min(n, e + ctx)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0, help="use the first N articles that are at least 2*core long")
    ap.add_argument("--core", type=int, default=192, help="fixed core length")
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 14, 31, 63], help="context on each side")
    ap.add_argument("--seam_margin", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    keep = [i for i, c in enumerate(chars) if len(c) >= 2 * args.core]  # at least one internal boundary
    if args.limit:
        keep = keep[:args.limit]
    chars, gold = [chars[i] for i in keep], [gold[i] for i in keep]
    lengths = [len(c) for c in chars]
    print(f"fixed core: core={args.core}, articles={len(chars)}", flush=True)

    logits_fn = make_logits_fn(load_finetuned(args.bert_dir, args.ckpt, device))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    g = np.concatenate(gold)
    offs = np.concatenate([[0], np.cumsum(lengths)])
    bounds = [int(offs[a]) + b for a, n in enumerate(lengths) for b in core_boundaries(n, args.core)]

    result = {"config": {"core": args.core, "contexts": args.contexts, "split": args.split,
                         "n_articles": len(chars), "n_boundaries": len(bounds),
                         "seam_margin": args.seam_margin}, "results": []}
    for ctx in args.contexts:
        windows = fixed_core_windows(lengths, args.core, ctx)
        labels, stats = decode_parallel(chars, logits_fn, *special, device, window=args.core + 2 * ctx,
                                        overlap=0, batch_size=args.batch_size, windows=windows,
                                        profile="hard", mask_left=ctx, mask_right=ctx)
        p = np.concatenate(labels)
        seam = seam_f1_fixed_positions(g, p, bounds, margin=args.seam_margin)
        overall = punctuation_prf(g, p, 8, 0)
        result["results"].append({"context": ctx, "window_size": args.core + 2 * ctx,
                                  "seam_f1": seam["seam_f1"], "interior_f1": seam["interior_f1"],
                                  "seam_support": seam["seam_support"], "overall_f1": overall["micro_f1"],
                                  "ter": stats.ter, "n_windows": stats.n_windows})
        print(f"ctx={ctx}: seamF1={seam['seam_f1']:.4f} interiorF1={seam['interior_f1']:.4f} "
              f"overall={overall['micro_f1']:.4f} TER={stats.ter:.3f}", flush=True)
        dump(args.out, result)


if __name__ == "__main__":
    main()
