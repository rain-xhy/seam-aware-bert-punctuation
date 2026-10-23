"""对两个方法进行配对bootstrap显著性检验。

从保存的预测结果中读取两个方法的逐篇F1，计算95%置信区间，
并进行配对bootstrap检验判断差异是否显著。

    python -m scripts.bootstrap_compare --result results/E4_dev.json \
        --preds results/E4_dev_preds.npz --data_dir prepared_data_complete \
        --method_a parallel --method_b serial_teacher --n_bootstrap 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.bootstrap import bootstrap_f1_ci, compute_per_article_f1, paired_bootstrap_test  # noqa: E402
from punct.runner import load_split  # noqa: E402


def split_by_article(pred: np.ndarray, lengths: list) -> list:
    """将连续的预测标签按文章切分"""
    offset = 0
    out = []
    for n in lengths:
        out.append(pred[offset:offset + n])
        offset += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--method_a", required=True)
    ap.add_argument("--method_b", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)
    _, gold = load_split(args.data_dir, result["split"])
    gold = gold[: result["n_articles"]]
    lengths = [len(g) for g in gold]

    npz = np.load(args.preds)
    pred_a = split_by_article(npz[args.method_a], lengths)
    pred_b = split_by_article(npz[args.method_b], lengths)

    f1_a = compute_per_article_f1(gold, pred_a)
    f1_b = compute_per_article_f1(gold, pred_b)

    ci_a = bootstrap_f1_ci(gold, pred_a, n_bootstrap=args.n_bootstrap, confidence=args.confidence)
    ci_b = bootstrap_f1_ci(gold, pred_b, n_bootstrap=args.n_bootstrap, confidence=args.confidence)
    test_result = paired_bootstrap_test(gold, pred_a, pred_b, n_bootstrap=args.n_bootstrap,
                                       confidence=args.confidence)

    print(f"=== Bootstrap比较: {args.method_a} vs {args.method_b} ===\n")
    print(f"{args.method_a}:")
    print(f"  均值F1: {f1_a.mean():.4f}")
    print(f"  {int(args.confidence*100)}% CI: [{ci_a['ci_lower']:.4f}, {ci_a['ci_upper']:.4f}]")
    print(f"  标准误: {ci_a['std_error']:.4f}\n")

    print(f"{args.method_b}:")
    print(f"  均值F1: {f1_b.mean():.4f}")
    print(f"  {int(args.confidence*100)}% CI: [{ci_b['ci_lower']:.4f}, {ci_b['ci_upper']:.4f}]")
    print(f"  标准误: {ci_b['std_error']:.4f}\n")

    print(f"配对检验:")
    print(f"  差值: {test_result['delta']:.4f}")
    print(f"  {int(args.confidence*100)}% CI: [{test_result['ci_lower']:.4f}, {test_result['ci_upper']:.4f}]")
    print(f"  显著优于: {test_result['superior']}")
    print(f"  p值(近似): {1 - test_result['confidence']:.4f}")


if __name__ == "__main__":
    main()
