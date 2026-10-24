"""Random position offset training (完善方案.md §4.3 item 5, conditional).

During training a crop normally starts at position id 0. This hook makes a
fraction of the training forward passes use position ids that start at a
random offset, so that position embeddings beyond the crop length are also
trained. It is installed as a forward pre-hook, so the training loop itself
does not change. Evaluation (model.eval()) is never affected.
"""
from __future__ import annotations

import random

import torch


def install_position_offset(model, prob: float, max_positions: int = 512, seed: int = 0):
    """Returns the hook handle. prob <= 0 installs nothing (returns None)."""
    if prob <= 0:
        return None
    rng = random.Random(seed)

    def pre_hook(module, args, kwargs):
        if not module.training or rng.random() >= prob:
            return None
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            return None
        batch, seq = ids.shape
        room = max_positions - seq
        if room <= 0:
            return None
        offset = rng.randint(0, room)
        kwargs = dict(kwargs)
        kwargs["position_ids"] = (torch.arange(seq, device=ids.device) + offset).unsqueeze(0).expand(batch, -1)
        return args, kwargs

    return model.register_forward_pre_hook(pre_hook, with_kwargs=True)
