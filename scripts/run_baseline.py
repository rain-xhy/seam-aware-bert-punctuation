"""频率基线实验：每个字符预测其在训练集中最常见的后续标点。

这是最简单的基线方法，不使用任何上下文信息，仅基于字符频率统计。

    python -m scripts.run_baseline --data_dir prepared_data_complete \
        --out results/baseline_frequency.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.metrics import fit_char_frequency_baseline, predict_char_frequency_baseline  # noqa: E402
from punct.runner import load_split, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("加载数据...")
    train_chars, train_labels = load_split(args.data_dir, "train")
    dev_chars, dev_labels = load_split(args.data_dir, "dev")
    test_chars, test_labels = load_split(args.data_dir, "test")

    # 训练频率表
    print("\n训练频率基线...")
    freq_table = fit_char_frequency_baseline(train_chars, train_labels, num_labels=8)
    print(f"  频率表覆盖 {len(freq_table)} 个字符")

    # 在验证集上预测
    print("\n验证集评估...")
    dev_pred = [predict_char_frequency_baseline(freq_table, chars) for chars in dev_chars]
    dev_quality = score(dev_labels, dev_pred)

    # 在测试集上预测
    print("测试集评估...")
    test_pred = [predict_char_frequency_baseline(freq_table, chars) for chars in test_chars]
    test_quality = score(test_labels, test_pred)

    result = {
        "method": "char_frequency_baseline",
        "description": "每个字符预测训练集中最常见的后续标点",
        "freq_table_size": len(freq_table),
        "dev": {
            "micro_f1": dev_quality["micro_f1"],
            "macro_f1": dev_quality["macro_f1"],
            "accuracy": dev_quality["accuracy"],
            "sentence_final_f1": dev_quality["sentence_final_f1"],
            "errors": dev_quality["errors"],
        },
        "test": {
            "micro_f1": test_quality["micro_f1"],
            "macro_f1": test_quality["macro_f1"],
            "accuracy": test_quality["accuracy"],
            "sentence_final_f1": test_quality["sentence_final_f1"],
            "errors": test_quality["errors"],
        },
    }

    print(f"\n验证集: F1={dev_quality['micro_f1']:.4f}")
    print(f"测试集: F1={test_quality['micro_f1']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
