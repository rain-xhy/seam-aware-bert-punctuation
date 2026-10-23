"""在验证集上训练温度校准参数，然后在测试集上评估。

Temperature scaling: 将模型logits除以温度T，并加上per-class bias，
在验证集上用Adam优化使预测概率更准确 (最小化交叉熵)。

    python -m scripts.run_calibration --ckpt checkpoints/T1_varlen/best.pt \
        --cal_split dev --test_split test --out results/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.calibrate import apply_calibration, calibrate_temperature_and_bias  # noqa: E402
from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import LogitsFn, decode_parallel, load_finetuned  # noqa: E402
from punct.runner import dump, load_split, score  # noqa: E402


def collect_logits_and_labels(articles, gold, logits_fn, special, device, window=400, overlap=100):
    """在验证集上收集所有logits和标签，用于训练校准参数"""
    cls_id, sep_id, pad_id = special
    _, stats, probs = decode_parallel(articles, logits_fn, cls_id, sep_id, pad_id, device,
                                     window=window, overlap=overlap, return_probs=True)
    # probs: list of [len, 8] per article
    all_probs = np.concatenate(probs)  # [total, 8]
    all_gold = np.concatenate(gold)    # [total]
    return all_probs, all_gold


def make_calibrated_logits_fn(model, temperature: float, bias: np.ndarray, use_bf16: bool = True) -> LogitsFn:
    """返回一个logits_fn，输出已校准的logits"""
    def fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if use_bf16 and input_ids.is_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            return apply_calibration(logits, temperature, bias)
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--cal_split", default="dev", help="calibration split")
    ap.add_argument("--test_split", default="test", help="evaluation split")
    ap.add_argument("--cal_limit", type=int, default=0, help="use first N articles for calibration")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    model = load_finetuned(args.bert_dir, args.ckpt, device)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    # 1. 收集校准集logits
    print(f"收集校准集 ({args.cal_split}) logits...")
    cal_chars, cal_gold = load_split(args.data_dir, args.cal_split)
    if args.cal_limit:
        cal_chars, cal_gold = cal_chars[:args.cal_limit], cal_gold[:args.cal_limit]

    def raw_logits_fn(ids, mask):
        with torch.inference_mode():
            return model(input_ids=ids, attention_mask=mask).logits

    probs, labels = collect_logits_and_labels(cal_chars, cal_gold, raw_logits_fn, special, device,
                                             args.window, args.overlap)
    # 转回logits (log概率)
    logits = torch.from_numpy(np.log(probs + 1e-12)).to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    # 2. 训练温度和bias
    print(f"训练温度校准参数 ({args.steps} steps)...")
    temperature, bias = calibrate_temperature_and_bias(logits, labels_t, n_classes=8,
                                                       steps=args.steps, lr=args.lr)
    print(f"  温度T = {temperature:.4f}")
    print(f"  bias = {bias}")

    # 3. 在测试集上评估
    print(f"\n评估测试集 ({args.test_split})...")
    test_chars, test_gold = load_split(args.data_dir, args.test_split)

    # 未校准
    _, _, probs_raw = decode_parallel(test_chars, raw_logits_fn, *special, device,
                                     window=args.window, overlap=args.overlap, return_probs=True)
    pred_raw = [p.argmax(-1) for p in probs_raw]
    score_raw = score(test_gold, pred_raw)

    # 已校准
    cal_logits_fn = make_calibrated_logits_fn(model, temperature, bias)
    _, _, probs_cal = decode_parallel(test_chars, cal_logits_fn, *special, device,
                                     window=args.window, overlap=args.overlap, return_probs=True)
    pred_cal = [p.argmax(-1) for p in probs_cal]
    score_cal = score(test_gold, pred_cal)

    result = {
        "calibration": {
            "temperature": float(temperature),
            "bias": bias.tolist(),
            "split": args.cal_split,
            "n_samples": int(logits.shape[0]),
        },
        "uncalibrated": {
            "micro_f1": score_raw["micro_f1"],
            "macro_f1": score_raw["macro_f1"],
            "accuracy": score_raw["accuracy"],
        },
        "calibrated": {
            "micro_f1": score_cal["micro_f1"],
            "macro_f1": score_cal["macro_f1"],
            "accuracy": score_cal["accuracy"],
        },
        "improvement": {
            "micro_f1": score_cal["micro_f1"] - score_raw["micro_f1"],
            "macro_f1": score_cal["macro_f1"] - score_raw["macro_f1"],
        },
    }

    print(f"\n未校准: F1={score_raw['micro_f1']:.4f}")
    print(f"已校准: F1={score_cal['micro_f1']:.4f} (Δ{result['improvement']['micro_f1']:+.4f})")

    dump(args.out, result)


if __name__ == "__main__":
    main()
