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


def seam_f1(
    gold: np.ndarray,
    pred: np.ndarray,
    windows,  # List[Window] from punct.inference
    seam_margin: int = 8,
    num_labels: int = 8,
    none_label: int = 0
) -> Dict:
    """计算接缝F1 vs 内部F1.

    Args:
        gold, pred: [text_len]
        windows: 窗口列表
        seam_margin: 接缝定义：离边缘≤seam_margin字

    Returns:
        {'seam_f1': ..., 'interior_f1': ..., 'seam_support': ..., 'interior_support': ...}
    """
    text_len = len(gold)

    # 标记接缝区域
    seam_mask = np.zeros(text_len, dtype=bool)
    for win in windows:
        # 左接缝
        for i in range(win.start, min(win.start + seam_margin, win.end)):
            if 0 <= i < text_len:
                seam_mask[i] = True
        # 右接缝
        for i in range(max(win.end - seam_margin, win.start), win.end):
            if 0 <= i < text_len:
                seam_mask[i] = True

    # 分组
    seam_gold = gold[seam_mask]
    seam_pred = pred[seam_mask]
    interior_gold = gold[~seam_mask]
    interior_pred = pred[~seam_mask]

    # 计算F1
    seam_prf = punctuation_prf(seam_gold, seam_pred, num_labels, none_label) if len(seam_gold) > 0 else {'micro_f1': 0.0}
    interior_prf = punctuation_prf(interior_gold, interior_pred, num_labels, none_label) if len(interior_gold) > 0 else {'micro_f1': 0.0}

    return {
        'seam_f1': seam_prf['micro_f1'],
        'interior_f1': interior_prf['micro_f1'],
        'seam_support': int(np.sum(seam_gold > 0)),
        'interior_support': int(np.sum(interior_gold > 0)),
        'seam_positions': int(np.sum(seam_mask)),
        'interior_positions': int(np.sum(~seam_mask))
    }


def f1_by_article_length(
    gold_per_article: List[np.ndarray],
    pred_per_article: List[np.ndarray],
    num_labels: int = 8,
    none_label: int = 0
) -> Dict:
    """按文章长度分组计算F1.

    Args:
        gold_per_article: [num_articles][article_len]
        pred_per_article: [num_articles][article_len]

    Returns:
        {'short': {'f1': ..., 'support': ...}, 'medium': ..., 'long': ...}
    """
    buckets = {'short': (0, 400), 'medium': (401, 2000), 'long': (2001, 999999)}
    results = {}

    for bucket_name, (min_len, max_len) in buckets.items():
        gold_list = []
        pred_list = []

        for g, p in zip(gold_per_article, pred_per_article):
            if min_len <= len(g) <= max_len:
                gold_list.append(g)
                pred_list.append(p)

        if len(gold_list) > 0:
            gold_concat = np.concatenate(gold_list)
            pred_concat = np.concatenate(pred_list)
            prf = punctuation_prf(gold_concat, pred_concat, num_labels, none_label)
            results[bucket_name] = {
                'micro_f1': prf['micro_f1'],
                'macro_f1': prf['macro_f1'],
                'support': prf['n_positions'],
                'num_articles': len(gold_list)
            }
        else:
            results[bucket_name] = {'micro_f1': 0.0, 'support': 0, 'num_articles': 0}

    return results


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


# ---------------------------------------------------------------------------
# Additional diagnostics (error decomposition, fixed-position seam F1,
# per-character frequency baseline).
# ---------------------------------------------------------------------------

def error_decomposition(gold: np.ndarray, pred: np.ndarray, none_label: int = 0) -> Dict:
    """Split every wrong position into one of three kinds.

    missed:   gold is a mark, prediction is "none"      (漏标)
    spurious: gold is "none", prediction is a mark      (误插)
    confused: both are marks but different classes      (类别错)
    """
    if gold.shape != pred.shape:
        raise ValueError(f"shape mismatch: gold={gold.shape} pred={pred.shape}")
    g_mark = gold != none_label
    p_mark = pred != none_label
    missed = int(np.sum(g_mark & ~p_mark))
    spurious = int(np.sum(~g_mark & p_mark))
    confused = int(np.sum(g_mark & p_mark & (gold != pred)))
    total = missed + spurious + confused
    return {
        "missed": missed,
        "spurious": spurious,
        "confused": confused,
        "total_errors": total,
        "gold_marks": int(g_mark.sum()),
    }


def seam_f1_fixed_positions(
    gold: np.ndarray,
    pred: np.ndarray,
    boundaries,
    margin: int = 8,
    num_labels: int = 8,
    none_label: int = 0,
) -> Dict:
    """Seam F1 with a method-independent definition of "seam".

    `boundaries` is a list of global character offsets (e.g. every multiple of
    a reference stride). A position counts as seam if it lies within `margin`
    characters of any boundary, on either side. Because the set does not
    depend on the decoding method, all methods are scored on the same
    positions.
    """
    n = len(gold)
    if gold.shape != pred.shape:
        raise ValueError(f"shape mismatch: gold={gold.shape} pred={pred.shape}")
    mask = np.zeros(n, dtype=bool)
    for b in boundaries:
        lo, hi = max(0, int(b) - margin), min(n, int(b) + margin)
        if lo < hi:
            mask[lo:hi] = True
    out = {"seam_positions": int(mask.sum()), "interior_positions": int((~mask).sum())}
    for name, m in (("seam", mask), ("interior", ~mask)):
        if m.any():
            prf = punctuation_prf(gold[m], pred[m], num_labels, none_label)
            out[f"{name}_f1"] = prf["micro_f1"]
        else:
            out[f"{name}_f1"] = 0.0
        out[f"{name}_support"] = int(np.sum(gold[m] != none_label))
    return out


def fit_char_frequency_baseline(char_ids_list, label_ids_list, num_labels: int = 8) -> Dict[int, int]:
    """For every character id, the most frequent label that follows it in the
    training data. Ties break toward the smaller label id."""
    counts: Dict[int, np.ndarray] = {}
    for chars, labels in zip(char_ids_list, label_ids_list):
        for c, l in zip(chars.tolist(), labels.tolist()):
            if l < 0:
                continue
            arr = counts.get(c)
            if arr is None:
                arr = counts[c] = np.zeros(num_labels, dtype=np.int64)
            arr[l] += 1
    return {c: int(np.argmax(a)) for c, a in counts.items()}


def predict_char_frequency_baseline(table: Dict[int, int], chars: np.ndarray, default: int = 0) -> np.ndarray:
    """Look every character up in the majority-label table."""
    return np.array([table.get(int(c), default) for c in chars], dtype=np.int64)
