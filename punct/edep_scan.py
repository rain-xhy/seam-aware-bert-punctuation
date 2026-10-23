"""EDEP-L / EDEP-R measured with the batched engine.

For every window of the plan, the model's own (unmerged) prediction at each
position is compared with the gold label, and the error is filed under the
position's distance to the window's left edge (EDEP-L) or right edge
(EDEP-R). An edge that coincides with the article's true start or end is
skipped: nothing is cut off there, so it is not a seam.

The masking widths m_l and m_r are the first distances at which the error
rate falls back to the centre-of-window error rate plus a tolerance.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from punct.edep import find_masking_threshold
from punct.engine import LogitsFn, PooledWindow, _run_batch, pool_windows


def edep_scan(
    articles: Sequence[Sequence[int]], gold: Sequence[np.ndarray], logits_fn: LogitsFn,
    cls_id: int, sep_id: int, pad_id: int, device: torch.device,
    window: int = 400, overlap: int = 200, tail: str = "short",
    max_distance: int = 100, batch_size: int = 8, tolerance: float = 0.005,
    windows: Optional[Sequence[PooledWindow]] = None,
) -> Dict:
    """Returns {'left': rates, 'right': rates, 'left_n': counts, 'right_n': counts,
    'centre_error', 'm_l', 'm_r'}; rates[d] is the error rate at distance d."""
    lengths = [len(a) for a in articles]
    pooled = sorted(windows if windows is not None else pool_windows(lengths, window, overlap, tail),
                    key=lambda w: w.end - w.start)
    err = {"left": np.zeros(max_distance + 1), "right": np.zeros(max_distance + 1)}
    cnt = {"left": np.zeros(max_distance + 1), "right": np.zeros(max_distance + 1)}
    centre_err = centre_n = 0

    bs = batch_size
    i = 0
    while i < len(pooled):
        batch = pooled[i:i + bs]
        chunks = [articles[w.article][w.start:w.end] for w in batch]
        try:
            probs = _run_batch(logits_fn, chunks, cls_id, sep_id, pad_id, device)
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1)
            continue
        pred = probs.argmax(-1).cpu().numpy()
        for b, w in enumerate(batch):
            n = w.end - w.start
            wrong = pred[b, 1:1 + n] != np.asarray(gold[w.article][w.start:w.end])
            if w.start > 0:  # internal left edge
                k = min(n, max_distance + 1)
                err["left"][:k] += wrong[:k]
                cnt["left"][:k] += 1
            if w.end < lengths[w.article]:  # internal right edge
                k = min(n, max_distance + 1)
                err["right"][:k] += wrong[::-1][:k]
                cnt["right"][:k] += 1
            lo, hi = n // 3, 2 * n // 3
            centre_err += int(wrong[lo:hi].sum())
            centre_n += hi - lo
        i += len(batch)

    left = err["left"] / np.maximum(cnt["left"], 1)
    right = err["right"] / np.maximum(cnt["right"], 1)
    centre = centre_err / max(centre_n, 1)
    d = np.arange(max_distance + 1)
    return {
        "left": left.tolist(), "right": right.tolist(),
        "left_n": cnt["left"].tolist(), "right_n": cnt["right"].tolist(),
        "centre_error": centre,
        "m_l": find_masking_threshold(d, left, centre, tolerance),
        "m_r": find_masking_threshold(d, right, centre, tolerance),
        "window": window, "overlap": overlap, "tail": tail, "max_distance": max_distance,
    }
