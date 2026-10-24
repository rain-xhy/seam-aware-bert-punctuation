"""Pipelined two-pass decoding.

decode_two_pass (two_pass.py) finishes pass 1 for every article before it
plans a single patch window. Here the work is one FIFO queue holding both
pass-1 windows and patch windows. As soon as the last pass-1 window of an
article has been processed, that article's merged probabilities are final:
its low-confidence seam positions are selected, patch windows are planned and
appended to the queue, where they share batches with pass-1 windows of
articles that have not finished yet. Pass 2 therefore does not wait for a
full extra round over the whole corpus.

The selection and patch rules are the same functions as in two_pass.py, so
the labels must equal decode_two_pass(..., select="confidence") exactly; the
test suite checks this. Random selection needs a corpus-wide budget and is
not supported here.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from punct.engine import (DecodeStats, LogitsFn, _encode_batch, _forward_encoded, edge_weights, pool_windows)
from punct.two_pass import PatchWindow, plan_patches, seam_candidates


def decode_two_pass_pipelined(
    articles: Sequence[Sequence[int]], logits_fn: LogitsFn, special: Tuple[int, int, int], device: torch.device,
    window: int = 400, overlap: int = 100, tail: str = "short", mask_left: int = 0, mask_right: int = 0,
    profile: str = "hard", batch_size: int = 8, seam_margin: int = 16, conf_thr: float = 0.9,
    patch_window: int = 400, edge_margin: int = 128, anchor_range: int = 16,
    calibration: Optional[Tuple[float, np.ndarray]] = None, num_labels: int = 8, edep=None,
):
    """-> (labels per article, info dict). Same arguments as decode_two_pass
    (confidence selection only)."""
    t_all = time.perf_counter()
    cls_id, sep_id, pad_id = special
    lengths = [len(a) for a in articles]
    pooled = pool_windows(lengths, window, overlap, tail)
    masks = seam_candidates(pooled, lengths, seam_margin)
    remaining = [0] * len(articles)
    for w in pooled:
        remaining[w.article] += 1

    prob_sum: Dict[int, torch.Tensor] = {}
    w_sum: Dict[int, torch.Tensor] = {}
    out: List[Optional[np.ndarray]] = [None] * len(articles)
    for a, n in enumerate(lengths):
        if n == 0:
            out[a] = np.zeros(0, dtype=np.int64)
        else:
            prob_sum[a] = torch.zeros(n, num_labels, dtype=torch.float32, device=device)
            w_sum[a] = torch.zeros(n, dtype=torch.float32, device=device)

    queue = deque(("w", w) for w in pooled)
    info = {"pass1_tokens": sum(w.end - w.start for w in pooled), "pass1_windows": len(pooled),
            "n_selected": 0, "n_patch_windows": 0, "patch_tokens": 0, "n_changed": 0,
            "n_oom_retries": 0, "n_forward_calls": 0, "forward_seconds": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def finalize(a: int) -> None:
        if bool((w_sum[a] <= 0).any().item()):
            raise ValueError("some positions received zero merge weight; mask_left+mask_right too large")
        merged = (prob_sum[a] / w_sum[a].clamp_min(1e-12)[:, None]).cpu().numpy()
        del prob_sum[a], w_sum[a]
        labels = merged.argmax(axis=1)
        out[a] = labels.copy()
        sel_probs = merged
        if calibration is not None:
            from punct.calibrate import calibrate_probs
            sel_probs = calibrate_probs(merged, calibration[0], calibration[1])
        idx = np.flatnonzero((sel_probs.max(axis=1) < conf_thr) & masks[a])
        if idx.size == 0:
            return
        info["n_selected"] += int(idx.size)
        patches = plan_patches({a: idx}, lengths, {a: labels}, patch_window, edge_margin, anchor_range)
        info["n_patch_windows"] += len(patches)
        for p in patches:
            info["patch_tokens"] += p.end - p.start
            queue.append(("p", p))

    bs = batch_size
    while queue:
        take = [queue.popleft() for _ in range(min(bs, len(queue)))]
        chunks = [articles[it.article][it.start:it.end] for _, it in take]
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = _forward_encoded(logits_fn, _encode_batch(chunks, cls_id, sep_id, pad_id), device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            info["forward_seconds"] += time.perf_counter() - t0
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            for it in reversed(take):
                queue.appendleft(it)
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1)
            info["n_oom_retries"] += 1
            continue
        info["n_forward_calls"] += 1
        pred = None
        for row, (kind, it) in enumerate(take):
            n = it.end - it.start
            if kind == "w":
                a = it.article
                wv = torch.from_numpy(edge_weights(n, it.start == 0, it.end == lengths[a], mask_left, mask_right,
                                                   profile, edep)).to(device)
                prob_sum[a][it.start:it.end] += probs[row, 1:1 + n] * wv[:, None]
                w_sum[a][it.start:it.end] += wv
                remaining[a] -= 1
                if remaining[a] == 0:
                    finalize(a)
            else:
                if pred is None:
                    pred = probs.argmax(-1).cpu().numpy()
                for pos in it.positions:
                    lab = int(pred[row, 1 + pos - it.start])
                    if out[it.article][pos] != lab:
                        info["n_changed"] += 1
                    out[it.article][pos] = lab

    total = max(sum(lengths), 1)
    info["ter"] = (info["pass1_tokens"] + info["patch_tokens"]) / total
    info["batch_size_used"] = bs
    if device.type == "cuda":
        torch.cuda.synchronize()
        info["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    info["total_seconds"] = time.perf_counter() - t_all
    return out, info
