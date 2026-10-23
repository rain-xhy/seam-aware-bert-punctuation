"""Evaluation metrics for the 8-class punctuation task.

This first version covers what train.py needs during training (micro/macro
P/R/F1 over the 7 pause-mark classes, label 0 = no punctuation excluded from
the "positive" set the way every classmate's report also does it — see
同学方案分析.md §2.3). Seam F1, length-bucket F1 and bootstrap intervals
(完善方案.md §6) are added in a later pass once the inference pipeline
produces per-article boundaries to bucket by.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix(gold: np.ndarray, pred: np.ndarray, num_labels: int) -> np.ndarray:
    """rows = gold label, cols = predicted label."""
    if gold.shape != pred.shape:
        raise ValueError(f"shape mismatch: gold={gold.shape} pred={pred.shape}")
    cm = np.zeros((num_labels, num_labels), dtype=np.int64)
    for g, p in zip(gold, pred):
        cm[g, p] += 1
    return cm


def punctuation_prf(gold: np.ndarray, pred: np.ndarray, num_labels: int, none_label: int = 0) -> Dict:
    """Per-class P/R/F1 (class `none_label` excluded from the punctuation
    classes but still part of the confusion matrix), plus micro and macro
    F1 over the punctuation classes only, plus overall accuracy."""
    cm = confusion_matrix(gold, pred, num_labels)
    punct_labels = [c for c in range(num_labels) if c != none_label]

    per_class = {}
    tp_sum = fp_sum = fn_sum = 0
    f1s = []
    for c in punct_labels:
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[c, :].sum())}
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        f1s.append(f1)

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    accuracy = float(np.trace(cm)) / cm.sum() if cm.sum() > 0 else 0.0

    return {
        "per_class": per_class,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": accuracy,
        "n_positions": int(cm.sum()),
    }


def sentence_final_f1(gold: np.ndarray, pred: np.ndarray, sentence_final_labels) -> Dict:
    """Binary F1 treating any of `sentence_final_labels` as the positive class
    (句末 F1: 。？！ merged, 完善方案.md §6)."""
    final_set = set(sentence_final_labels)
    gold_bin = np.isin(gold, list(final_set))
    pred_bin = np.isin(pred, list(final_set))
    tp = int(np.sum(gold_bin & pred_bin))
    fp = int(np.sum(~gold_bin & pred_bin))
    fn = int(np.sum(gold_bin & ~pred_bin))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": int(gold_bin.sum())}


def all_none_baseline_accuracy(gold: np.ndarray, none_label: int = 0) -> float:
    """Accuracy of always predicting 'no punctuation' — the trivial baseline
    every classmate reports alongside their real F1."""
    return float(np.mean(gold == none_label))


def compute_metrics(eval_pred) -> Dict:
    """Compute metrics for training/validation. `eval_pred` is an EvalPrediction
    object with .predictions (logits) and .label_ids (gold labels)."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # logits shape: (batch, seq_len, num_labels)
    # labels shape: (batch, seq_len)
    pred = np.argmax(logits, axis=-1).flatten()
    gold = labels.flatten()

    # Filter out ignored positions (label == -100)
    mask = gold != -100
    pred = pred[mask]
    gold = gold[mask]

    # Compute punctuation P/R/F1 (7 classes, excluding label 0)
    prf = punctuation_prf(gold, pred, num_labels=8, none_label=0)

    # Sentence-final F1 (labels 2=。, 3=?, 4=!)
    sf = sentence_final_f1(gold, pred, sentence_final_labels=[2, 3, 4])

    return {
        "micro_f1": prf["micro_f1"],
        "macro_f1": prf["macro_f1"],
        "micro_precision": prf["micro_precision"],
        "micro_recall": prf["micro_recall"],
        "sentence_final_f1": sf["f1"],
        "accuracy": prf["accuracy"],
    }
