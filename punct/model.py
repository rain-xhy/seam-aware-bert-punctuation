"""BERT + linear token-classification head for the 8-class punctuation task.

Thin wrapper around transformers.BertForTokenClassification so the rest of
the codebase (train.py, infer_*.py) does not depend on the exact
transformers API. Also provides the encoder-weight check the plan calls
for (§3.1): comparing a fixed slice of the encoder's own weights before
and after training, to catch the case where only the new classification
head was ever updated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import BertForTokenClassification

NUM_LABELS = 8
WEIGHT_CHECK_PATH = "bert.encoder.layer.0.attention.self.query.weight"


def load_model(model_dir: str | Path, num_labels: int = NUM_LABELS) -> BertForTokenClassification:
    """Load bert-base-chinese (or a fine-tuned checkpoint saved the same way)
    with a num_labels-way token-classification head, from local files only."""
    return BertForTokenClassification.from_pretrained(
        str(model_dir), num_labels=num_labels, local_files_only=True
    )


def save_model(model: BertForTokenClassification, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))


def get_weight_check_tensor(model: BertForTokenClassification, path: str = WEIGHT_CHECK_PATH) -> torch.Tensor:
    """Fetch one named parameter's value, detached and on CPU, for later comparison."""
    state = dict(model.named_parameters())
    if path not in state:
        raise KeyError(f"{path!r} not found among model parameters; available keys start with: "
                        + ", ".join(sorted(state.keys())[:5]))
    return state[path].detach().clone().cpu()


def encoder_weights_changed(before: torch.Tensor, after: torch.Tensor, atol: float = 1e-6) -> Tuple[bool, float]:
    """(changed, max_abs_diff). changed is False only if every element of
    `after` is within atol of the corresponding element of `before` — i.e.
    training never touched this encoder weight, which would mean only the
    new classification head was updated."""
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: before={before.shape} after={after.shape}")
    diff = (after - before).abs()
    max_diff = float(diff.max().item())
    return max_diff > atol, max_diff
