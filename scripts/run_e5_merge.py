"""E5实验：对比三种合并方式 (hard mask, gaussian, EDEP-weighted)。

EDEP-weighted: 每个位置的权重 = 1 - 该位置离最近边缘d的错误率。
需要先跑EDEP扫描得到错误率曲线，然后用它作为位置权重。

    python -m scripts.run_e5_merge --ckpt checkpoints/T1_varlen/best.pt \
        --split dev --limit 300 --window 400 --overlap 100 \
        --out results/E5_merge_comparison.json
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
from punct.edep_scan import edep_scan  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import dump, load_split, run_method, score  # noqa: E402


def edep_weight_profile(left_err: list, right_err: list) -> callable:
    """返回一个权重函数: 基于EDEP错误率的位置加权"""
    left_err = np.array(left_err)
    right_err = np.array(right_err)

    def weight_fn(length: int, at_article_start: bool, at_article_end: bool,
                  mask_left: int, mask_right: int) -> np.ndarray:
        """每个位置的权重 = 1 - min(左边缘错误率, 右边缘错误率)"""
        w = np.ones(length, dtype=np.float32)
        for i in range(length):
            # 离左边缘的距离
            d_left = i if not at_article_start else 999
            # 离右边缘的距离
            d_right = length - 1 - i if not at_article_end else 999

            # 取两个方向中较近的错误率
            err_left = left_err[min(d_left, len(left_err) - 1)] if d_left < len(left_err) else 0
            err_right = right_err[min(d_right, len(right_err) - 1)] if d_right < len(right_err) else 0
            err = max(err_left, err_right)  # 取更差的方向

            w[i] = max(1.0 - err, 0.1)  # 最低0.1避免完全屏蔽
        return w

    return weight_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    if args.limit:
        chars, gold = chars[:args.limit], gold[:args.limit]

    model = load_finetuned(args.bert_dir, args.ckpt, device)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    # 1. 先扫描EDEP
    print("扫描EDEP-L/R曲线...")
    edep_result = edep_scan(chars, gold, logits_fn, *special, device,
                           window=args.window, overlap=args.overlap, batch_size=args.batch_size)
    print(f"  EDEP-L: m_l={edep_result['m_l']}, 中心错误率={edep_result['centre_error']:.4f}")
    print(f"  EDEP-R: m_r={edep_result['m_r']}")

    # 2. 测试三种合并方式
    configs = [
        ("hard", dict(profile="hard", mask_left=edep_result['m_l'], mask_right=edep_result['m_r'])),
        ("gauss", dict(profile="gauss", mask_left=0, mask_right=0)),
        # 经验权重: 1 - EDEP错误率, 直接走engine的"edep"权重
        ("edep", dict(profile="edep", mask_left=0, mask_right=0,
                      edep=(edep_result["left"], edep_result["right"]))),
    ]

    result = {
        "config": {"window": args.window, "overlap": args.overlap, "split": args.split},
        "edep": edep_result,
        "methods": {}
    }

    print(f"\n测试三种合并方式:")
    for name, kw in configs:
        print(f"  {name}...", end=" ", flush=True)
        pred, stats = run_method("parallel", chars, logits_fn, special, device,
                                window=args.window, overlap=args.overlap, batch_size=args.batch_size,
                                **kw)
        q = score(gold, pred)
        result["methods"][name] = {
            "config": {k: v for k, v in kw.items() if k != "edep"},
            "quality": {"micro_f1": q["micro_f1"], "macro_f1": q["macro_f1"]},
            "cost": {"ter": stats["ter"], "seconds": stats["seconds"]},
        }
        print(f"F1={q['micro_f1']:.4f}, TER={stats['ter']:.3f}")

    dump(args.out, result)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
