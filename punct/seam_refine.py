"""Seam补窗细化 (完善方案.md §4.5, optional refinement). After the main
parallel inference pass, this module identifies positions near window seams
(±10 chars of every 300-char boundary) where the model's confidence was low
(e.g. max_prob < 0.7), re-runs a SINGLE centered window at each such
position, and replaces the low-confidence prediction with the new one.

identify_low_confidence_seam_positions scans logits (or predicted probs) at
seam-zone positions and returns a list of (article_idx, char_pos) tuples
where confidence was below threshold.

refine_seam_positions takes those positions, constructs one centered window
per position (完善方案.md §4.5: window_size=400, the position itself at
offset 200), runs model.forward on each window, and replaces the label at
that position with the new prediction.

完善方案.md §6's ablation table reports F1 with and without seam refinement.
If the gain is <0.5 pp and the overhead is >10% of inference time, refinement
is dropped from the final method (完善方案.md explicitly says "如果提升不显著,
则不在最终方法中使用").
"""
from typing import List, Sequence, Tuple

import numpy as np
import torch


def identify_low_confidence_seam_positions(
    articles_logits: Sequence[torch.Tensor],
    windows: List,  # List[WindowSpec]
    confidence_threshold: float = 0.7,
    seam_margin: int = 10,
) -> List[Tuple[int, int]]:
    """Scan positions within ±`seam_margin` chars of every window seam (where
    one window's responsibility zone ends and another's starts). Return a list
    of (article_idx, char_pos) where max_prob < confidence_threshold. windows
    must be sorted by (article_idx, responsible_start). articles_logits[i] is
    shape [n_chars_in_article_i, n_classes]."""
    low_conf_positions = []
    sorted_windows = sorted(windows, key=lambda w: (w.article_idx, w.responsible_start))
    for i in range(len(sorted_windows) - 1):
        w1, w2 = sorted_windows[i], sorted_windows[i + 1]
        if w1.article_idx != w2.article_idx:
            continue
        seam_pos = w1.responsible_end  # == w2.responsible_start by construction
        logits = articles_logits[w1.article_idx]
        for offset in range(-seam_margin, seam_margin + 1):
            pos = seam_pos + offset
            if 0 <= pos < logits.shape[0]:
                probs = torch.softmax(logits[pos], dim=0)
                max_prob = probs.max().item()
                if max_prob < confidence_threshold:
                    low_conf_positions.append((w1.article_idx, pos))
    return low_conf_positions


def refine_seam_positions(
    low_conf_positions: List[Tuple[int, int]],
    articles_char_ids: Sequence[Sequence[int]],
    articles_labels: List[np.ndarray],
    model: torch.nn.Module,
    codec,  # CharCodec with cls_id, sep_id
    device: torch.device,
    window_size: int = 400,
) -> int:
    """For each (article_idx, pos) in low_conf_positions, construct a window
    centered at pos (size `window_size`), run model.forward, and replace
    articles_labels[article_idx][pos] with the new prediction. Returns the
    number of positions actually refined (some may be skipped if they're too
    close to article boundaries to center a full window)."""
    model.eval()
    refined_count = 0
    with torch.no_grad():
        for art_idx, pos in low_conf_positions:
            char_ids = articles_char_ids[art_idx]
            half = window_size // 2
            start = max(0, pos - half)
            end = min(len(char_ids), pos + half)
            if end - start < 50:  # window too short to be useful
                continue
            chunk = char_ids[start:end]
            ids = [codec.cls_id] + list(chunk) + [codec.sep_id]
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(ids_t)  # [1, seq, 8]
            preds = logits[0, 1:-1].argmax(dim=-1).cpu().numpy()
            offset_in_chunk = pos - start
            if 0 <= offset_in_chunk < len(preds):
                articles_labels[art_idx][pos] = preds[offset_in_chunk]
                refined_count += 1
    return refined_count
