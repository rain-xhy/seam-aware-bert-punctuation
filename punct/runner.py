"""Experiment runner helpers: load a prepared split, run one decoding method
with a loaded model, and score the predictions with the full metric set.

Every method returns (labels per article, stats dict) so the experiment
scripts can treat serial and parallel decoders uniformly.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from punct.engine import LogitsFn, decode_parallel
from punct.infer_serial import cross_article_serial, tang_serial_v1, tang_serial_v2, teacher_serial
from punct.metrics import (
    all_none_baseline_accuracy,
    confusion_matrix,
    error_decomposition,
    punctuation_prf,
    sentence_final_f1,
    seam_f1_fixed_positions,
)

SENT_FINAL = (2, 3, 4)


def load_split(data_dir: str, split: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    chars = np.load(os.path.join(data_dir, f"{split}_articles_char.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, f"{split}_articles_label.npy"), allow_pickle=True)
    return [np.asarray(c, dtype=np.int64) for c in chars], [np.asarray(l, dtype=np.int64) for l in labels]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _single_model_fn(logits_fn: LogitsFn, cls_id, sep_id, device) -> Callable:
    def fn(chunk):
        ids = torch.tensor([[cls_id] + list(map(int, chunk)) + [sep_id]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        return logits_fn(ids, mask)[0, 1:-1].argmax(-1).cpu().numpy()
    return fn


def _batch_model_fn(logits_fn: LogitsFn, cls_id, sep_id, pad_id, device) -> Callable:
    def fn(chunks):
        m = max(len(c) for c in chunks) + 2
        ids = np.full((len(chunks), m), pad_id, dtype=np.int64)
        mask = np.zeros((len(chunks), m), dtype=np.int64)
        for i, c in enumerate(chunks):
            n = len(c)
            ids[i, 0], ids[i, 1:1 + n], ids[i, 1 + n] = cls_id, np.asarray(c, dtype=np.int64), sep_id
            mask[i, :n + 2] = 1
        lg = logits_fn(torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)).argmax(-1).cpu().numpy()
        return [lg[i, 1:1 + len(c)] for i, c in enumerate(chunks)]
    return fn


def run_method(name: str, articles: Sequence[np.ndarray], logits_fn: LogitsFn, special: Tuple[int, int, int],
               device: torch.device, **kw) -> Tuple[List[np.ndarray], Dict]:
    """name in: direct_cut, serial_yu (W=64, keep 2 sentences), serial_tang (W=125),
    serial_teacher (400/300), serial_cross (400/300, cross-article batch), parallel."""
    cls_id, sep_id, pad_id = special
    total = int(sum(len(a) for a in articles))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()
    if name == "parallel":
        labels, st = decode_parallel(articles, logits_fn, cls_id, sep_id, pad_id, device, **kw)
        stats = {"seconds": st.total_seconds, "forward_seconds": st.forward_seconds, "ter": st.ter,
                 "n_windows": st.n_windows, "n_forward_calls": st.n_forward_calls,
                 "n_oom_retries": st.n_oom_retries, "peak_mem_mb": st.peak_mem_mb}
    elif name == "direct_cut":
        labels, st = decode_parallel(articles, logits_fn, cls_id, sep_id, pad_id, device,
                                     window=kw.get("window", 64), overlap=0, tail="short",
                                     batch_size=kw.get("batch_size", 16))
        stats = {"seconds": st.total_seconds, "forward_seconds": st.forward_seconds, "ter": st.ter,
                 "n_windows": st.n_windows, "n_forward_calls": st.n_forward_calls,
                 "n_oom_retries": st.n_oom_retries, "peak_mem_mb": st.peak_mem_mb}
    elif name == "serial_cross":
        fn = _batch_model_fn(logits_fn, cls_id, sep_id, pad_id, device)
        out, calls = cross_article_serial([a.tolist() for a in articles], fn, window=400, commit_prefix=300)
        labels = [np.asarray(o, dtype=np.int64) for o in out]
        _sync(device)
        stats = {"n_forward_calls": calls}
    else:
        fn = _single_model_fn(logits_fn, cls_id, sep_id, device)
        labels, calls, fwd = [], 0, 0
        for a in articles:
            ids = a.tolist()
            if name == "serial_teacher":
                out, c = teacher_serial(ids, fn, window=400, commit_prefix=300)
            elif name == "serial_yu":
                out, c = tang_serial_v1(ids, fn, window=64, keep_sentences=2)
            elif name == "serial_tang":
                out, c = tang_serial_v2(ids, fn, window=125)
            else:
                raise ValueError(f"unknown method {name!r}")
            labels.append(np.asarray(out, dtype=np.int64))
            calls += c
        _sync(device)
        stats = {"n_forward_calls": calls}
    _sync(device)
    stats["seconds"] = time.perf_counter() - t0
    stats["chars_per_second"] = total / max(stats["seconds"], 1e-9)
    stats["text_chars"] = total
    if device.type == "cuda":
        stats["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return labels, stats


def score(gold: Sequence[np.ndarray], pred: Sequence[np.ndarray], seam_stride: int = 400,
          seam_margin: int = 8) -> Dict:
    """Full quality metric set on concatenated articles (+ per-article F1 summary + length groups)."""
    g, p = np.concatenate(gold), np.concatenate(pred)
    prf = punctuation_prf(g, p, 8, 0)
    out = {
        "micro_f1": prf["micro_f1"], "micro_precision": prf["micro_precision"],
        "micro_recall": prf["micro_recall"], "macro_f1": prf["macro_f1"], "accuracy": prf["accuracy"],
        "per_class": {str(k): v for k, v in prf["per_class"].items()},
        "sentence_final_f1": sentence_final_f1(g, p, SENT_FINAL)["f1"],
        "all_none_accuracy": all_none_baseline_accuracy(g),
        "errors": error_decomposition(g, p),
        "confusion": confusion_matrix(g, p, 8).tolist(),
    }
    # fixed-position seams: every `seam_stride` chars inside each article
    offs = np.concatenate([[0], np.cumsum([len(x) for x in gold])])
    bounds = [int(offs[i]) + b for i, x in enumerate(gold) for b in range(seam_stride, len(x), seam_stride)]
    out["seam_fixed"] = seam_f1_fixed_positions(g, p, bounds, margin=seam_margin)
    per_art = []
    for ga, pa in zip(gold, pred):
        if (ga != 0).any() or (pa != 0).any():
            per_art.append(punctuation_prf(ga, pa, 8, 0)["micro_f1"])
    if per_art:
        out["per_article_f1_mean"] = float(np.mean(per_art))
        out["per_article_f1_median"] = float(np.median(per_art))
        out["n_articles_scored"] = len(per_art)

    # F1 by article length groups (short: <=400, medium: 401-2000, long: >2000)
    from punct.metrics import f1_by_article_length
    out["f1_by_length"] = f1_by_article_length(gold, pred, num_labels=8, none_label=0)

    return out


def exact_agreement(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> Dict:
    """Character-level agreement between two decodings of the same texts."""
    x, y = np.concatenate(a), np.concatenate(b)
    return {"n": int(x.size), "n_diff": int((x != y).sum()), "agreement": float((x == y).mean())}


def dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
