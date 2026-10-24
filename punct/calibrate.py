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


def calibrate_probs(probs: np.ndarray, temperature: float, bias: np.ndarray) -> np.ndarray:
    """Calibrate already-softmaxed probabilities [N, C]: softmax((log p + b) / T).
    Used when only merged probabilities are available (two-pass selection)."""
    z = (np.log(np.clip(probs, 1e-12, 1.0)) + np.asarray(bias)) / max(temperature, 0.01)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """ECE over top-1 confidence with equal-width bins, plus per-bin data for a
    reliability diagram. probs [N, C], labels [N]."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels differ in length")
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    n = max(len(conf), 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        ece += cnt / n * abs(c - a)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": cnt,
                     "confidence": c, "accuracy": a})
    return {"ece": float(ece), "n": len(conf), "bins": bins}
