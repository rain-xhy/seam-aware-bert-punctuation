"""Compute window start/end positions for a fixed-stride parallel sliding window.

Pure functions only: the stride S = W - V is fixed before any model output
exists, so the full list of windows for an article of known length can be
produced in one shot and handed to a batcher. This module has no knowledge
of models, batching or merging; it only answers "where do the windows go".

Two tail rules (完善方案.md §4.2):
  "aligned" - the last window is pulled back so it ends exactly at the
              article's last character (length == W, like Tang 2021 and
              most classmates' implementations); this re-covers part of
              the previous window a second time.
  "short"   - the last window starts one stride after the previous window
              and is simply shorter than W; nothing before it is recovered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

TailRule = Literal["aligned", "short"]


@dataclass(frozen=True)
class Window:
    start: int  # inclusive, 0-based, into the article's character stream
    end: int    # exclusive


def plan_windows(length: int, window: int, overlap: int, tail: TailRule = "short") -> List[Window]:
    """Windows covering [0, length) with window size `window` and overlap `overlap`.

    Raises ValueError for a non-positive window or an overlap that is not
    strictly smaller than the window (overlap == window would never advance).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if overlap < 0 or overlap >= window:
        raise ValueError(f"overlap must satisfy 0 <= overlap < window, got {overlap} for window={window}")
    if tail not in ("aligned", "short"):
        raise ValueError(f"unknown tail rule: {tail!r}")
    if length <= 0:
        return []
    if length <= window:
        return [Window(0, length)]

    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    windows = [Window(s, s + window) for s in starts]
    last_end = windows[-1].end
    if last_end < length:
        if tail == "aligned":
            windows.append(Window(length - window, length))
        else:
            new_start = windows[-1].start + stride
            windows.append(Window(new_start, length))
    return windows


def coverage_ok(windows: List[Window], length: int) -> bool:
    """True if every position in [0, length) is covered by at least one window
    and no window goes out of [0, length) bounds."""
    if length <= 0:
        return len(windows) == 0
    covered = [False] * length
    for w in windows:
        if w.start < 0 or w.end > length or w.start >= w.end:
            return False
        for i in range(w.start, w.end):
            covered[i] = True
    return all(covered)


def total_forward_chars(windows: List[Window]) -> int:
    """Sum of window lengths; the numerator of the token-expansion ratio (TER)."""
    return sum(w.end - w.start for w in windows)
