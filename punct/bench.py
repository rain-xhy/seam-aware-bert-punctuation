"""Timing protocol and consistency checks shared by the efficiency experiments.

time_interleaved: warm-up once per configuration, then run the
configurations in rotating order `repeats` times (a laptop GPU throttles,
so a fixed order would favour whichever configuration runs first), and
report median / std / min / max of the wall time of each.

agreement: character-level agreement between two decodings, with the
number of differing characters (written to the log even when zero).
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_interleaved(configs: Dict[str, Callable[[], object]], repeats: int = 5, warmup: int = 1) -> Dict[str, Dict]:
    """configs: name -> zero-argument callable doing the full job.
    Returns name -> {median, std, min, max, runs}. The result object of the
    last run is kept under 'last_result' so callers can score it."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    names = list(configs)
    last: Dict[str, object] = {}
    for n in names:
        for _ in range(warmup):
            last[n] = configs[n]()
    runs: Dict[str, List[float]] = {n: [] for n in names}
    for r in range(repeats):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            _sync()
            t0 = time.perf_counter()
            last[n] = configs[n]()
            _sync()
            runs[n].append(time.perf_counter() - t0)
    out = {}
    for n in names:
        v = runs[n]
        out[n] = {
            "median": statistics.median(v),
            "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "runs": v,
            "last_result": last[n],
        }
    return out


def agreement(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> Dict:
    x, y = np.concatenate(a), np.concatenate(b)
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} vs {y.shape}")
    nd = int((x != y).sum())
    return {"n": int(x.size), "n_diff": nd, "agreement": 1.0 - nd / max(x.size, 1)}
