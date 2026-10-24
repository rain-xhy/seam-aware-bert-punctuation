"""Coarser label schemes (完善方案.md §9 optional: 3-class / 2-class labels).

The 8-class head is kept (labels 0..7). A coarse scheme only changes which
label ids occur: 3-class uses {0: none, 1: inner mark, 2: sentence-final},
2-class uses {0: none, 1: any mark}. Ids >= K are never produced as gold, so
the unused outputs are simply pushed down during training. The same mapping
is applied to gold (and to 8-class predictions) at evaluation time, so a
coarse model and a coarsened 8-class model are scored on the same footing.

Label ids follow prepare_complete_data.LABELS = [O, ， 。 ？ ！ ； ： 、].
"""
from __future__ import annotations

from typing import Dict

import numpy as np

IGNORE_INDEX = -100

SCHEMES: Dict[str, np.ndarray] = {
    "8": np.arange(8, dtype=np.int64),
    "3": np.array([0, 1, 2, 2, 2, 1, 1, 1], dtype=np.int64),
    "2": np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
}


def coarsen(labels, scheme: str):
    """Map 8-class label ids to `scheme`. Works on numpy arrays and torch
    tensors; IGNORE_INDEX (-100) is kept as is."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown label scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    table = SCHEMES[scheme]
    if hasattr(labels, "detach"):  # torch tensor
        import torch
        t = torch.as_tensor(table, device=labels.device)
        keep = labels == IGNORE_INDEX
        out = t[labels.clamp(min=0)]
        return torch.where(keep, torch.full_like(out, IGNORE_INDEX), out)
    arr = np.asarray(labels)
    out = table[np.clip(arr, 0, 7)]
    return np.where(arr == IGNORE_INDEX, IGNORE_INDEX, out)
