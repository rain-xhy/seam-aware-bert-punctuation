"""多窗口子集评测：只在>=W字的文章上评测，避免短文偏差。

短文章(<=W)只需要一个窗口，不涉及接缝合并，评测时应该分开报告。
这个脚本会输出"全集"和"多窗口子集"两个结果。

    python -m scripts.eval_multiwindow_subset --result results/E4_dev.json \
        --preds results/E4_dev_preds.npz --data_dir prepared_data_complete \
        --window 400 --out results/E4_dev_multiwindow.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.metrics import punctuation_prf  # noqa: E402
from punct.runner import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="原始结果JSON")
    ap.add_argument("--preds", required=True, help="预测结果NPZ")
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--window", type=int, required=True, help="窗口大小W")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)

    _, gold = load_split(args.data_dir, result["split"])
    gold = gold[: result["n_articles"]]
    lengths = np.array([len(g) for g in gold])

    npz = np.load(args.preds)

    # 分组：单窗口文章 vs 多窗口文章
    single_window_idx = lengths <= args.window
    multi_window_idx = lengths > args.window

    print(f"文章总数: {len(gold)}")
    print(f"  单窗口 (<=W={args.window}): {single_window_idx.sum()}")
    print(f"  多窗口 (>W): {multi_window_idx.sum()}")

    output = {
        "config": {"window": args.window, "split": result["split"]},
        "stats": {
            "n_total": len(gold),
            "n_single_window": int(single_window_idx.sum()),
            "n_multi_window": int(multi_window_idx.sum()),
        },
        "methods": {}
    }

    for method_name in npz.files:
        pred_all = npz[method_name]
        gold_all = np.concatenate(gold)

        # 全集
        prf_all = punctuation_prf(gold_all, pred_all, 8, 0)

        # 分组评测
        offset = 0
        gold_single, pred_single = [], []
        gold_multi, pred_multi = [], []

        for i, g in enumerate(gold):
            n = len(g)
            p = pred_all[offset:offset + n]
            if single_window_idx[i]:
                gold_single.append(g)
                pred_single.append(p)
            else:
                gold_multi.append(g)
                pred_multi.append(p)
            offset += n

        prf_single = punctuation_prf(np.concatenate(gold_single), np.concatenate(pred_single), 8, 0)
        prf_multi = punctuation_prf(np.concatenate(gold_multi), np.concatenate(pred_multi), 8, 0)

        output["methods"][method_name] = {
            "all": {
                "micro_f1": prf_all["micro_f1"],
                "macro_f1": prf_all["macro_f1"],
                "accuracy": prf_all["accuracy"],
            },
            "single_window": {
                "micro_f1": prf_single["micro_f1"],
                "macro_f1": prf_single["macro_f1"],
                "accuracy": prf_single["accuracy"],
            },
            "multi_window": {
                "micro_f1": prf_multi["micro_f1"],
                "macro_f1": prf_multi["macro_f1"],
                "accuracy": prf_multi["accuracy"],
            },
        }

        print(f"\n{method_name}:")
        print(f"  全集: F1={prf_all['micro_f1']:.4f}")
        print(f"  单窗口: F1={prf_single['micro_f1']:.4f}")
        print(f"  多窗口: F1={prf_multi['micro_f1']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
