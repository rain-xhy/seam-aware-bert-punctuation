"""Position weight profiles for probability merging.

Three strategies:
- hard: binary mask (0 within mask_left/mask_right of edges, 1 elsewhere)
- gauss: Gaussian centered on window midpoint
- edep: data-driven weights from EDEP error curves (1 - error_rate)
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np


def hard_profile(mask_left: int, mask_right: int) -> Callable:
    """Binary mask: 0 within mask_left/mask_right of internal edges, 1 elsewhere."""
    def weight_fn(length: int, at_article_start: bool, at_article_end: bool) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)
        if not at_article_start and mask_left > 0:
            w[:mask_left] = 0.0
        if not at_article_end and mask_right > 0:
            w[max(length - mask_right, 0):] = 0.0
        return w
    return weight_fn


def gauss_profile(sigma_ratio: float = 0.25) -> Callable:
    """Gaussian centered on window midpoint."""
    def weight_fn(length: int, at_article_start: bool, at_article_end: bool) -> np.ndarray:
        centre = (length - 1) / 2.0
        sigma = max(sigma_ratio * length, 1e-6)
        positions = np.arange(length)
        return np.exp(-0.5 * ((positions - centre) / sigma) ** 2).astype(np.float32)
    return weight_fn


def edep_profile(left_errors: List[float], right_errors: List[float],
                 floor: float = 0.1, ceiling: float = 1.0) -> Callable:
    """EDEP-driven weights: weight = ceiling - error_rate, clamped to [floor, ceiling].

    Args:
        left_errors: EDEP-L curve (error rate by distance from left edge)
        right_errors: EDEP-R curve (error rate by distance from right edge)
        floor: minimum weight (avoid complete masking)
        ceiling: maximum weight
    """
    left_err = np.array(left_errors, dtype=np.float32)
    right_err = np.array(right_errors, dtype=np.float32)

    def weight_fn(length: int, at_article_start: bool, at_article_end: bool) -> np.ndarray:
        w = np.full(length, ceiling, dtype=np.float32)

        # Apply left edge penalty
        if not at_article_start:
            for i in range(min(length, len(left_err))):
                err = left_err[i]
                w[i] = np.clip(ceiling - err, floor, ceiling)

        # Apply right edge penalty
        if not at_article_end:
            for i in range(min(length, len(right_err))):
                err = right_err[i]
                pos = length - 1 - i
                # Take the worse (higher error) of the two directions
                w[pos] = min(w[pos], np.clip(ceiling - err, floor, ceiling))

        return w

    return weight_fn


def linear_decay_profile(threshold_left: int, threshold_right: int) -> Callable:
    """Linear decay from edge to threshold, then constant 1.0."""
    def weight_fn(length: int, at_article_start: bool, at_article_end: bool) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)

        if not at_article_start and threshold_left > 0:
            for i in range(min(threshold_left, length)):
                w[i] = i / threshold_left

        if not at_article_end and threshold_right > 0:
            for i in range(min(threshold_right, length)):
                pos = length - 1 - i
                w[pos] = min(w[pos], i / threshold_right)

        return w

    return weight_fn
