"""显示典型的预测错误样例，用于误差分析。

从保存的预测结果中找出错误位置，按错误类型分组（漏标、误插、类别错），
展示前后文上下文，便于定性分析模型的弱点。

    python -m scripts.show_errors --result results/E4_dev.json \
        --preds results/E4_dev_preds.npz --data_dir prepared_data_complete \
        --method parallel --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.runner import load_split  # noqa: E402

LABEL_NAMES = ["", "，", "。", "？", "！", "、", "；", "："]


def find_errors(gold: np.ndarray, pred: np.ndarray, limit: int = 50) -> dict:
    """分类错误：missed(漏标), spurious(误插), confused(类别错)"""
    errors = {"missed": [], "spurious": [], "confused": []}
    for i in range(len(gold)):
        if gold[i] == pred[i]:
            continue
        if gold[i] != 0 and pred[i] == 0:
            errors["missed"].append(i)
        elif gold[i] == 0 and pred[i] != 0:
            errors["spurious"].append(i)
        else:
            errors["confused"].append(i)
        if sum(len(v) for v in errors.values()) >= limit:
            break
    return errors


def show_context(chars: List[np.ndarray], pos: int, window: int = 10) -> Tuple[str, int]:
    """返回 (上下文字符串, 错误位置在字符串中的索引)"""
    offset = 0
    for art_idx, art in enumerate(chars):
        if pos < offset + len(art):
            local = pos - offset
            lo = max(0, local - window)
            hi = min(len(art), local + window + 1)
            return "".join(chr(c) for c in art[lo:hi]), local - lo
        offset += len(art)
    return "", 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--method", default="parallel")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--window", type=int, default=12, help="context window around error")
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, labels = load_split(args.data_dir, result["split"])
    chars = chars[: result["n_articles"]]
    labels = labels[: result["n_articles"]]
    gold = np.concatenate(labels)
    pred = np.load(args.preds)[args.method]

    errors = find_errors(gold, pred, args.limit)
    print(f"=== 误差样例分析 ({args.method}, {result['split']} split) ===\n")

    for kind, positions in errors.items():
        if not positions:
            continue
        print(f"\n## {kind.upper()} ({len(positions)} 个样例):\n")
        for idx in positions[:10]:  # 每类最多显示10个
            ctx, local_pos = show_context(chars, idx, args.window)
            g_label = LABEL_NAMES[gold[idx]] or "无"
            p_label = LABEL_NAMES[pred[idx]] or "无"

            before = ctx[:local_pos]
            after = ctx[local_pos + 1:] if local_pos + 1 < len(ctx) else ""
            char = ctx[local_pos] if local_pos < len(ctx) else "?"

            print(f"  位置 {idx}: [{before}]{char}[{after}]")
            print(f"    金标: {g_label}  预测: {p_label}")
            print()


if __name__ == "__main__":
    main()
