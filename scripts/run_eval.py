"""Evaluation script: run one or more inference methods on val or test set,
compute F1 (overall, per-class, seam-zone, sentence-final), report wall-clock
time and forward-call counts (完善方案.md §6). Usage:

    python -m scripts.run_eval T1 --methods teacher_serial parallel --split val
    python -m scripts.run_eval T1 --methods all --split test

Saves results to logs/eval_<checkpoint_name>_<split>.json. Every method's
entry includes:
  - micro_f1, macro_f1, sentence_final_f1, per_class P/R/F1
  - seam_f1 (±10 chars around every 300-char seam, 完善方案.md §6)
  - wall_seconds, n_forward_calls, chars_per_second

Methods:
  teacher_serial       - 老师的 400/300 串行 (infer_serial.teacher_serial)
  tang_v1              - 俞敬松 2019 decoding rule (infer_serial.tang_serial_v1)
  tang_v2              - Tang 2021 improved serial (infer_serial.tang_serial_v2)
  cross_article_serial - 老师规则 + 跨文章批量 (infer_serial.cross_article_serial)
  parallel             - 不对称屏蔽 + 短尾窗 + center_priority merge
  all                  - run all of the above

The 'parallel' method uses window_plan.plan_windows + infer_parallel.batch_
inference_parallel + merge.center_priority + infer_parallel.apply_short_tail_
windows. This is 完善方案.md's main proposed method.
"""
import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from punct.char_codec import CharCodec
from punct.infer_parallel import (
    WindowSpec,
    apply_short_tail_windows,
    batch_inference_parallel,
)
from punct.infer_serial import (
    cross_article_serial,
    tang_serial_v1,
    tang_serial_v2,
    teacher_serial,
)
from punct.merge import center_priority
from punct.metrics import confusion_matrix, punctuation_prf
from punct.model import load_model
from punct.window_plan import plan_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_articles(path: str, codec: CharCodec):
    """Load .npz file produced by data_prep.py, decode char_ids and labels
    back to lists. Returns (articles_char_ids, articles_labels)."""
    data = np.load(path, allow_pickle=True)
    articles_char_ids = [arr.tolist() for arr in data["char_ids"]]
    articles_labels = [arr.tolist() for arr in data["labels"]]
    return articles_char_ids, articles_labels


def seam_zone_f1(pred: Sequence[int], gold: Sequence[int], windows: List[WindowSpec], margin: int = 10) -> float:
    """F1 computed only on positions within ±`margin` chars of a window seam
    (完善方案.md §6, "接缝 F1"). A seam is any position where one window's
    responsibility zone ends and another's starts (i.e. window.responsible_end
    == next_window.responsible_start). If no seams exist, return 1.0."""
    seam_positions = set()
    sorted_windows = sorted(windows, key=lambda w: (w.article_idx, w.responsible_start))
    for i in range(len(sorted_windows) - 1):
        if sorted_windows[i].article_idx == sorted_windows[i + 1].article_idx:
            seam_pos = sorted_windows[i].responsible_end
            for offset in range(-margin, margin + 1):
                p = seam_pos + offset
                if 0 <= p < len(pred):
                    seam_positions.add(p)
    if not seam_positions:
        return 1.0
    seam_pred = [pred[i] for i in sorted(seam_positions)]
    seam_gold = [gold[i] for i in sorted(seam_positions)]
    cm = confusion_matrix(seam_pred, seam_gold, n_classes=8)
    return punctuation_prf(cm)["micro_f1"]


def eval_teacher_serial(
    articles_char_ids, articles_labels, model, codec, device
) -> Dict:
    """Run infer_serial.teacher_serial on every article, concatenate predictions
    and gold, compute F1. Return dict with F1 metrics + wall_seconds +
    n_forward_calls."""

    def model_fn(char_ids):
        ids_t = torch.tensor([[codec.cls_id] + list(char_ids) + [codec.sep_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t)
        preds = logits[0, 1:-1].argmax(dim=-1).cpu().numpy()  # skip [CLS] and [SEP]
        return preds

    all_pred = []
    all_gold = []
    total_calls = 0
    start = time.perf_counter()
    for char_ids, labels in zip(articles_char_ids, articles_labels):
        pred_labels, calls = teacher_serial(char_ids, model_fn, window=400, commit_prefix=300)
        all_pred.extend(pred_labels)
        all_gold.extend(labels)
        total_calls += calls
    wall = time.perf_counter() - start

    cm = confusion_matrix(all_pred, all_gold, n_classes=8)
    metrics = punctuation_prf(cm)
    metrics["wall_seconds"] = wall
    metrics["n_forward_calls"] = total_calls
    metrics["chars_per_second"] = len(all_pred) / wall if wall > 0 else 0
    return metrics


def eval_tang_v1(articles_char_ids, articles_labels, model, codec, device) -> Dict:
    def model_fn(char_ids):
        ids_t = torch.tensor([[codec.cls_id] + list(char_ids) + [codec.sep_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t)
        return logits[0, 1:-1].argmax(dim=-1).cpu().numpy()

    all_pred = []
    all_gold = []
    total_calls = 0
    start = time.perf_counter()
    for char_ids, labels in zip(articles_char_ids, articles_labels):
        pred_labels, calls = tang_serial_v1(char_ids, model_fn, window=64, keep_sentences=2)
        all_pred.extend(pred_labels)
        all_gold.extend(labels)
        total_calls += calls
    wall = time.perf_counter() - start

    cm = confusion_matrix(all_pred, all_gold, n_classes=8)
    metrics = punctuation_prf(cm)
    metrics["wall_seconds"] = wall
    metrics["n_forward_calls"] = total_calls
    metrics["chars_per_second"] = len(all_pred) / wall if wall > 0 else 0
    return metrics


def eval_tang_v2(articles_char_ids, articles_labels, model, codec, device) -> Dict:
    def model_fn(char_ids):
        ids_t = torch.tensor([[codec.cls_id] + list(char_ids) + [codec.sep_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t)
        return logits[0, 1:-1].argmax(dim=-1).cpu().numpy()

    all_pred = []
    all_gold = []
    total_calls = 0
    start = time.perf_counter()
    for char_ids, labels in zip(articles_char_ids, articles_labels):
        pred_labels, calls = tang_serial_v2(char_ids, model_fn, window=125)
        all_pred.extend(pred_labels)
        all_gold.extend(labels)
        total_calls += calls
    wall = time.perf_counter() - start

    cm = confusion_matrix(all_pred, all_gold, n_classes=8)
    metrics = punctuation_prf(cm)
    metrics["wall_seconds"] = wall
    metrics["n_forward_calls"] = total_calls
    metrics["chars_per_second"] = len(all_pred) / wall if wall > 0 else 0
    return metrics


def eval_cross_article_serial(articles_char_ids, articles_labels, model, codec, device, batch_size=16) -> Dict:
    def batch_model_fn(chunks):
        max_len = max(len(c) for c in chunks) + 2  # +2 for [CLS] and [SEP]
        batch_ids = []
        for c in chunks:
            ids = [codec.cls_id] + list(c) + [codec.sep_id]
            ids += [codec.pad_id] * (max_len - len(ids))
            batch_ids.append(ids)
        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t)
        preds_list = []
        for i, c in enumerate(chunks):
            preds_list.append(logits[i, 1 : 1 + len(c)].argmax(dim=-1).cpu().numpy())
        return preds_list

    start = time.perf_counter()
    pred_labels_list, total_calls = cross_article_serial(
        articles_char_ids, batch_model_fn, window=400, commit_prefix=300
    )
    wall = time.perf_counter() - start

    all_pred = [lab for article_pred in pred_labels_list for lab in article_pred]
    all_gold = [lab for article_gold in articles_labels for lab in article_gold]
    cm = confusion_matrix(all_pred, all_gold, n_classes=8)
    metrics = punctuation_prf(cm)
    metrics["wall_seconds"] = wall
    metrics["n_forward_calls"] = total_calls
    metrics["chars_per_second"] = len(all_pred) / wall if wall > 0 else 0
    return metrics


def eval_parallel(
    articles_char_ids, articles_labels, model, codec, device, batch_size=16, left_base=0.1, right_base=0.5
) -> Dict:
    """完善方案.md's main proposed method: plan_windows + asymmetric_mask +
    short_tail_windows + batch_inference_parallel + center_priority merge."""
    # plan windows for all articles
    all_windows: List[WindowSpec] = []
    for art_idx, char_ids in enumerate(articles_char_ids):
        windows = plan_windows(len(char_ids), window=400, step=300)
        for w in windows:
            all_windows.append(WindowSpec(art_idx, w.start, w.end, w.responsible_start, w.responsible_end))

    # apply short-tail extension
    all_windows = apply_short_tail_windows(all_windows, articles_char_ids, tail_margin=50)

    # batch inference
    start = time.perf_counter()
    raw_labels_list, total_calls = batch_inference_parallel(
        all_windows, articles_char_ids, model, codec, device, batch_size, left_base, right_base
    )
    # raw_labels_list is "last window wins" by default; apply center_priority merge
    # (完善方案.md §4.2, center_priority is the main merge rule)
    final_labels_list = []
    for art_idx, raw_labels in enumerate(raw_labels_list):
        art_windows = [w for w in all_windows if w.article_idx == art_idx]
        merged = center_priority(raw_labels, art_windows)
        final_labels_list.append(merged)
    wall = time.perf_counter() - start

    all_pred = [lab for article_pred in final_labels_list for lab in article_pred]
    all_gold = [lab for article_gold in articles_labels for lab in article_gold]
    cm = confusion_matrix(all_pred, all_gold, n_classes=8)
    metrics = punctuation_prf(cm)
    metrics["wall_seconds"] = wall
    metrics["n_forward_calls"] = total_calls
    metrics["chars_per_second"] = len(all_pred) / wall if wall > 0 else 0
    # seam F1 (完善方案.md §6)
    metrics["seam_f1"] = seam_zone_f1(all_pred, all_gold, all_windows, margin=10)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="T1, T2, T3, T4, or T5 (subdirectory of checkpoints/)")
    parser.add_argument("--methods", nargs="+", default=["parallel"], help="teacher_serial, tang_v1, tang_v2, cross_article_serial, parallel, or 'all'")
    parser.add_argument("--split", default="val", help="val or test")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if "all" in args.methods:
        args.methods = ["teacher_serial", "tang_v1", "tang_v2", "cross_article_serial", "parallel"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec(os.path.join(ROOT, "bert-base-chinese", "vocab.txt"))
    ckpt_dir = os.path.join(ROOT, "checkpoints", args.checkpoint)
    model = load_model(ckpt_dir)
    model.to(device)
    model.eval()

    split_path = os.path.join(ROOT, "data", f"{args.split}.npz")
    articles_char_ids, articles_labels = load_articles(split_path, codec)

    results = {"checkpoint": args.checkpoint, "split": args.split, "methods": {}}

    for method in args.methods:
        print(f"Running {method}...", flush=True)
        if method == "teacher_serial":
            metrics = eval_teacher_serial(articles_char_ids, articles_labels, model, codec, device)
        elif method == "tang_v1":
            metrics = eval_tang_v1(articles_char_ids, articles_labels, model, codec, device)
        elif method == "tang_v2":
            metrics = eval_tang_v2(articles_char_ids, articles_labels, model, codec, device)
        elif method == "cross_article_serial":
            metrics = eval_cross_article_serial(articles_char_ids, articles_labels, model, codec, device, args.batch_size)
        elif method == "parallel":
            metrics = eval_parallel(articles_char_ids, articles_labels, model, codec, device, args.batch_size)
        else:
            raise ValueError(f"unknown method {method!r}")
        results["methods"][method] = metrics
        print(f"  micro_f1={metrics['micro_f1']:.4f}, wall={metrics['wall_seconds']:.2f}s, "
              f"calls={metrics['n_forward_calls']}, throughput={metrics['chars_per_second']:.0f} char/s")

    out_path = os.path.join(ROOT, "logs", f"eval_{args.checkpoint}_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
