"""固定核心隔离实验：固定核心长度，改变两侧上下文，测量对接缝F1的影响。

核心固定为192字，两侧上下文分别测试0/14/31/63字，观察上下文对接缝预测的影响。
这个实验不需要额外训练，用主模型在验证集上测即可。

    python -m scripts.run_fixed_core --ckpt checkpoints/T1_varlen/best.pt \
        --split dev --core 192 --contexts 0 14 31 63 \
        --out results/fixed_core_experiment.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import LogitsFn, PooledWindow, decode_parallel  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.metrics import seam_f1_fixed_positions  # noqa: E402
from punct.runner import dump, load_split  # noqa: E402


def fixed_core_windows(articles: Sequence[np.ndarray], core: int, context: int) -> List[PooledWindow]:
    """为每篇文章生成固定核心+上下文的窗口"""
    windows = []
    for a, art in enumerate(articles):
        n = len(art)
        if n < core:
            # 文章太短，整篇作为一个窗口
            windows.append(PooledWindow(a, 0, n))
            continue

        # 从文章中部取core字，两侧各加context字
        mid = n // 2
        core_start = mid - core // 2
        core_end = core_start + core

        # 扩展上下文
        start = max(0, core_start - context)
        end = min(n, core_end + context)

        windows.append(PooledWindow(a, start, end))

    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--core", type=int, default=192, help="固定核心长度")
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 14, 31, 63],
                   help="两侧上下文长度")
    ap.add_argument("--seam_margin", type=int, default=8, help="接缝区域半径")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)

    # 只选择长度>core的文章
    long_indices = [i for i, c in enumerate(chars) if len(c) >= args.core]
    if args.limit:
        long_indices = long_indices[:args.limit]

    chars = [chars[i] for i in long_indices]
    gold = [gold[i] for i in long_indices]

    print(f"固定核心实验: core={args.core}, 文章数={len(chars)}")

    model = load_finetuned(args.bert_dir, args.ckpt, device)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    result = {
        "config": {"core": args.core, "contexts": args.contexts, "split": args.split,
                   "n_articles": len(chars)},
        "results": []
    }

    # 确定接缝位置：每个窗口核心区域的左右边界
    gold_concat = np.concatenate(gold)
    offsets = np.concatenate([[0], np.cumsum([len(g) for g in gold])])

    for ctx in args.contexts:
        print(f"\n测试 context={ctx} (窗口={args.core + 2*ctx})...", end=" ", flush=True)

        windows = fixed_core_windows(chars, args.core, ctx)

        # 解码
        labels, stats = decode_parallel(chars, logits_fn, *special, device,
                                       batch_size=args.batch_size, windows=windows)
        pred_concat = np.concatenate(labels)

        # 计算接缝F1：核心区域的左右边界±margin
        seam_positions = []
        for w in windows:
            n = len(chars[w.article])
            # 核心区域在窗口中的相对位置
            core_start_local = ctx if ctx < w.start + ctx else w.start
            core_end_local = len(chars[w.article][w.start:w.end]) - ctx if w.end - ctx < n else len(chars[w.article][w.start:w.end])

            # 转全局位置
            global_start = int(offsets[w.article]) + w.start + core_start_local
            global_end = int(offsets[w.article]) + w.start + core_end_local

            # 左右边界作为接缝
            if w.start > 0:  # 非文章开头
                seam_positions.append(global_start)
            if w.end < n:    # 非文章结尾
                seam_positions.append(global_end)

        seam_result = seam_f1_fixed_positions(gold_concat, pred_concat, seam_positions,
                                             margin=args.seam_margin)

        # 整体F1
        from punct.metrics import punctuation_prf
        overall = punctuation_prf(gold_concat, pred_concat, 8, 0)

        result["results"].append({
            "context": ctx,
            "window_size": args.core + 2 * ctx,
            "seam_f1": seam_result["f1"],
            "seam_precision": seam_result["precision"],
            "seam_recall": seam_result["recall"],
            "overall_f1": overall["micro_f1"],
            "ter": stats.ter,
            "n_windows": stats.n_windows,
        })

        print(f"接缝F1={seam_result['f1']:.4f}, 整体F1={overall['micro_f1']:.4f}")

    dump(args.out, result)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
