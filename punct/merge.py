"""Merge per-window class probabilities back into one full-article distribution.

Three merge strategies (完善方案.md §4.2-§4.3, §7):
  tang_drop_last_sentence  - delete the last sentence of each window's own
                              prediction before concatenating (Tang 2021's
                              own method, §6.1); needs per-window predicted
                              labels, not just probabilities, so it is kept
                              separate from the probability-weighted path.
  hard_mask                - zero weight within m_l/m_r of each window's
                              edges (the article's true start/end are never
                              masked), then average the surviving probability
                              mass across all windows covering a position.
  position_weighted        - same idea but with a continuous weight profile
                              instead of a hard 0/1 mask (gaussian or an
                              empirical 1-error-rate profile, picked by E5).

This module implements only the CPU reference path. A GPU path using
index_add_ is added separately in infer_parallel.py once merge.py is
trusted; test_merge_gpu_matches_cpu there will compare the two bit-for-bit
on real tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from punct.window_plan import Window


@dataclass(frozen=True)
class WindowProbs:
    """One window's class probabilities, aligned to its own [0, end-start) axis."""

    window: Window
    probs: np.ndarray  # shape (end - start, num_labels), rows sum to 1


def merge_hard_mask(
    windows: Sequence[WindowProbs], article_length: int, num_labels: int,
    mask_left: int, mask_right: int,
) -> np.ndarray:
    """Average probabilities, excluding mask_left/mask_right chars at each
    window's own edges — except at the article's true start/end, which are
    never masked (there is no more context to gain by widening there)."""
    prob_sum = np.zeros((article_length, num_labels), dtype=np.float64)
    weight_sum = np.zeros(article_length, dtype=np.float64)
    for wp in windows:
        w, probs = wp.window, wp.probs
        local_len = w.end - w.start
        # a window's own left/right edge is only masked when it is an
        # internal seam, i.e. when this window does NOT itself start at
        # the article's true beginning (or end at the article's true end)
        window_touches_article_start = w.start == 0
        window_touches_article_end = w.end == article_length
        for local_i in range(local_len):
            global_i = w.start + local_i
            masked_left = (not window_touches_article_start) and local_i < mask_left
            masked_right = (not window_touches_article_end) and (local_len - 1 - local_i) < mask_right
            if masked_left or masked_right:
                continue
            prob_sum[global_i] += probs[local_i]
            weight_sum[global_i] += 1.0
    uncovered = weight_sum == 0
    if uncovered.any():
        raise ValueError(
            f"{uncovered.sum()} position(s) have zero weight after masking; "
            "mask_left/mask_right too large for the given overlap"
        )
    return prob_sum / weight_sum[:, None]


def merge_position_weighted(
    windows: Sequence[WindowProbs], article_length: int, num_labels: int,
    weight_fn: Callable[[int, int], float],
) -> np.ndarray:
    """Like merge_hard_mask but with a continuous weight_fn(local_i, window_len)
    instead of a 0/1 mask. weight_fn must return a value > 0 everywhere, or
    positions covered by only that window would get zero total weight."""
    prob_sum = np.zeros((article_length, num_labels), dtype=np.float64)
    weight_sum = np.zeros(article_length, dtype=np.float64)
    for wp in windows:
        w, probs = wp.window, wp.probs
        local_len = w.end - w.start
        for local_i in range(local_len):
            global_i = w.start + local_i
            weight = weight_fn(local_i, local_len)
            if weight <= 0:
                raise ValueError(f"weight_fn must be > 0 everywhere, got {weight}")
            prob_sum[global_i] += probs[local_i] * weight
            weight_sum[global_i] += weight
    return prob_sum / weight_sum[:, None]


def gaussian_weight(sigma_ratio: float = 0.25) -> Callable[[int, int], float]:
    """A weight_fn for merge_position_weighted: highest at the window centre,
    falling off as a Gaussian of the distance from centre. sigma_ratio scales
    the standard deviation relative to the window length."""

    def weight_fn(local_i: int, local_len: int) -> float:
        centre = (local_len - 1) / 2.0
        sigma = max(sigma_ratio * local_len, 1e-6)
        return float(np.exp(-0.5 * ((local_i - centre) / sigma) ** 2))

    return weight_fn


def merge_tang_drop_last_sentence(
    windows: Sequence[WindowProbs], article_length: int, num_labels: int,
    predicted_labels_by_window: Sequence[np.ndarray], sentence_final_label_ids: Sequence[int],
) -> np.ndarray:
    """Tang 2021's own merge rule: for each window except the last, find the
    last position (in local coordinates) whose predicted label is a sentence-
    final mark, drop everything from there to the window's end, and keep only
    the remaining prefix; concatenate the kept prefixes plus the final
    window's full prediction. Returns one-hot "probabilities" (this method
    only ever had hard labels, not a probability to average)."""
    onehot = np.zeros((article_length, num_labels), dtype=np.float64)
    filled = np.zeros(article_length, dtype=bool)
    for idx, wp in enumerate(windows):
        w = wp.window
        labels = predicted_labels_by_window[idx]
        is_last_window = idx == len(windows) - 1
        if is_last_window:
            keep_len = w.end - w.start
        else:
            cut_positions = [i for i, lab in enumerate(labels) if lab in sentence_final_label_ids]
            keep_len = (cut_positions[-1] + 1) if cut_positions else (w.end - w.start)
        for local_i in range(keep_len):
            global_i = w.start + local_i
            if filled[global_i]:
                continue
            onehot[global_i, labels[local_i]] = 1.0
            filled[global_i] = True
    if not filled.all():
        missing = int((~filled).sum())
        raise ValueError(f"{missing} position(s) left unfilled by merge_tang_drop_last_sentence")
    return onehot
