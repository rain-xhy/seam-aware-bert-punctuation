"""Parallel sliding window with the "drop the last sentence" merge (method A).

All windows of all articles are planned in advance and run as pooled batches,
exactly like the other parallel decoders. Only the merge differs: for every
window except the article's last one, everything after the window's last
predicted sentence-final mark is discarded, and each position takes the label
of the first window (in left-to-right order) that kept it.

If the dropped tail is longer than the overlap, some positions are kept by no
window. Those are filled from the first window that covers them and counted in
`n_gap_filled`, so the effect of the gap is visible in the logs.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from punct.engine import DecodeStats, LogitsFn, PooledWindow, _encode_batch, _forward_encoded, pool_windows

SENT_FINAL = (2, 3, 4)


def window_labels(articles: Sequence[Sequence[int]], pooled: Sequence[PooledWindow], logits_fn: LogitsFn,
                  special: Tuple[int, int, int], device: torch.device, batch_size: int = 16,
                  stats: Optional[DecodeStats] = None) -> List[np.ndarray]:
    """Argmax labels of every window (returned in the order of `pooled`),
    batched by length with OOM halve-and-retry."""
    cls_id, sep_id, pad_id = special
    order = sorted(range(len(pooled)), key=lambda k: pooled[k].end - pooled[k].start)
    out: List[Optional[np.ndarray]] = [None] * len(pooled)
    bs, i = batch_size, 0
    while i < len(order):
        idx = order[i:i + bs]
        chunks = [articles[pooled[k].article][pooled[k].start:pooled[k].end] for k in idx]
        try:
            t0 = time.perf_counter()
            probs = _forward_encoded(logits_fn, _encode_batch(chunks, cls_id, sep_id, pad_id), device)
            pred = probs.argmax(-1).cpu().numpy()
            if stats is not None:
                stats.forward_seconds += time.perf_counter() - t0
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1)
            if stats is not None:
                stats.n_oom_retries += 1
            continue
        if stats is not None:
            stats.n_forward_calls += 1
        for row, k in enumerate(idx):
            out[k] = pred[row, 1:1 + len(chunks[row])].astype(np.int64)
        i += len(idx)
    return out  # type: ignore[return-value]


def merge_drop_last_sentence(lengths: Sequence[int], pooled: Sequence[PooledWindow],
                             labels: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], int]:
    """-> (labels per article, number of positions filled without being kept)."""
    out = [np.zeros(n, dtype=np.int64) for n in lengths]
    filled = [np.zeros(n, dtype=bool) for n in lengths]
    by_article = {}
    for k, w in enumerate(pooled):
        by_article.setdefault(w.article, []).append(k)
    gap = 0
    for a, ks in by_article.items():
        ks.sort(key=lambda k: pooled[k].start)
        for pos_in_list, k in enumerate(ks):
            w, lab = pooled[k], labels[k]
            keep = len(lab)
            if pos_in_list < len(ks) - 1:
                hits = np.flatnonzero(np.isin(lab, SENT_FINAL))
                if hits.size:
                    keep = int(hits[-1]) + 1
            seg = slice(w.start, w.start + keep)
            todo = ~filled[a][seg]
            out[a][seg][todo] = lab[:keep][todo]
            filled[a][seg] = True
        for k in ks:  # positions no window kept: take the first covering window
            w, lab = pooled[k], labels[k]
            seg = slice(w.start, w.end)
            todo = ~filled[a][seg]
            if todo.any():
                gap += int(todo.sum())
                out[a][seg][todo] = lab[todo]
                filled[a][seg] = True
    return out, gap


def decode_tang_parallel(articles: Sequence[Sequence[int]], logits_fn: LogitsFn, special: Tuple[int, int, int],
                         device: torch.device, window: int = 125, overlap: int = 20, tail: str = "aligned",
                         batch_size: int = 16):
    """-> (labels per article, DecodeStats, info dict)."""
    t_all = time.perf_counter()
    lengths = [len(a) for a in articles]
    pooled = pool_windows(lengths, window, overlap, tail)
    stats = DecodeStats(n_windows=len(pooled), text_chars=sum(lengths),
                        forward_tokens=sum(w.end - w.start for w in pooled))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    labs = window_labels(articles, pooled, logits_fn, special, device, batch_size, stats)
    out, gap = merge_drop_last_sentence(lengths, pooled, labs)
    if device.type == "cuda":
        stats.peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20
    stats.total_seconds = time.perf_counter() - t_all
    return out, stats, {"n_gap_filled": gap}
