"""Post-hoc probability calibration (完善方案.md §4.4, optional refinement).

The model's raw logits may be miscalibrated: a predicted probability of 0.9
for PERIOD does not necessarily mean "90% chance this is actually a period."
Calibration fits a simple rescaling (temperature T and per-class bias b_c)
on a held-out calibration set (e.g. the first 20% of val articles) and
applies it at inference time to every window's logits BEFORE taking argmax.

calibrate_temperature_and_bias fits T and b_c by minimizing cross-entropy on
the calibration set. The calibrated logits are:

    calibrated_logits[c] = (raw_logits[c] + b_c) / T

apply_calibration applies the fitted T and b_c to new logits.

完善方案.md §6's ablation table reports F1 with and without calibration; if
the gain is <0.5 pp, calibration is dropped from the final method to keep
inference fast (no extra pass over the calibration set at training time).
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def calibrate_temperature_and_bias(
    logits: torch.Tensor, labels: torch.Tensor, n_classes: int = 8, lr: float = 0.01, steps: int = 100
) -> Tuple[float, np.ndarray]:
    """Fit temperature T and per-class bias b_c on a calibration set. logits
    shape [N, n_classes], labels shape [N] (int class ids). Returns (T, b)
    where b is shape [n_classes]."""
    device = logits.device
    T = torch.tensor(1.0, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([T, b], lr=lr)
    for _ in range(steps):
        calibrated = (logits + b) / T.clamp(min=0.01)  # avoid T=0
        loss = F.cross_entropy(calibrated, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return T.item(), b.detach().cpu().numpy()


def apply_calibration(logits: torch.Tensor, temperature: float, bias: np.ndarray) -> torch.Tensor:
    """Apply fitted T and b to new logits. logits shape [batch, seq, n_classes]
    or [seq, n_classes]. Returns calibrated logits of the same shape."""
    b_t = torch.tensor(bias, dtype=logits.dtype, device=logits.device)
    return (logits + b_t) / max(temperature, 0.01)
