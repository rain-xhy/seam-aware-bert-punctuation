"""Parallel inference with asymmetric masking and short-tail windows
(完善方案.md §4.2 核心改进1, §4.3 核心改进2). The key difference from
infer_serial.py: every article's window plan is FULLY DETERMINED ahead of
time (by window_plan.plan_windows), so all windows across all articles can
be batched together without waiting for predictions. The trade: a window
near a 300-char seam sees LESS left context than a 400-char centered
window would, and this shows up as lower F1 in the ±10 char接缝 F1 metric
compared to the teacher's serial method on short articles (完善方案.md §6
acknowledges this). The win: on long articles (2000+ chars) where most
windows are NOT near a seam, throughput goes up 4-5× and overall F1 stays
level or improves slightly.

asymmetric_mask builds the attention_mask for one window: positions
BEFORE the window's responsibility zone get masked out at a rate that
grows quadratically as you move left from the zone's left edge
(完善方案.md §4.2). Positions AFTER the zone get masked at a constant
rate (default 0.5). The zone itself is never masked. This forces the
model to weight the chars it's RESPONSIBLE for more than the left-context
padding, which is the opposite of what BERT's position embeddings naturally
do (nearby tokens matter more).

batch_inference_parallel takes the output of window_plan.plan_windows (a
list of window specs, each carrying start/end/responsible_start/responsible_end)
and runs them all in one or more GPU batches, applying asymmetric_mask to
every window. Returns (label per char, n_forward_calls). The caller
(scripts/run_eval.py) then picks labels via merge.py and computes F1.

short_tail_window (完善方案.md §4.3 核心改进2): when a window's right edge
would land within `tail_margin` chars of the article's true end, push the
window's right edge all the way to the end so the model sees the ACTUAL
final </s> token, not a fake mid-sentence cutoff. This recovers 2-3
percentage points of F1 on sentence-final marks in the last 50 chars of
every article, per 完善方案.md §6's ablation table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class WindowSpec:
    """One window's slice boundaries and responsibility zone, per
    window_plan.plan_windows output."""

    article_idx: int
    start: int  # char index in the article's char_ids where this window starts (inclusive)
    end: int  # char index where this window ends (exclusive)
    responsible_start: int  # first char index this window is responsible for labelling
    responsible_end: int  # one past the last char index this window labels


def asymmetric_mask(
    seq_len: int, responsible_start: int, responsible_end: int, left_base: float = 0.1, right_base: float = 0.5,
) -> torch.Tensor:
    """Attention mask (shape [seq_len]) for one window. seq_len includes [CLS]
    and [SEP], so responsible_start/end are shifted +1 when indexing into the
    mask. Positions inside [responsible_start, responsible_end) are NEVER
    masked (mask=1.0). Positions to the LEFT of responsible_start are masked
    with probability that grows quadratically as you move left (完善方案.md
    §4.2 formula). Positions to the RIGHT are masked at constant `right_base`.
    The [CLS] and [SEP] tokens themselves are never masked (always 1.0)."""
    mask = torch.ones(seq_len, dtype=torch.float32)
    resp_left = responsible_start + 1  # +1 because [CLS] is at index 0
    resp_right = responsible_end + 1
    zone_width = resp_right - resp_left

    # left side: quadratic drop from the zone's left edge
    for i in range(1, resp_left):  # skip [CLS] at 0
        dist = resp_left - i
        ratio = dist / max(zone_width, 1)
        mask[i] = max(0.0, 1.0 - left_base * (ratio**2))

    # right side: constant masking
    for i in range(resp_right, seq_len - 1):  # skip [SEP] at seq_len-1
        mask[i] = 1.0 - right_base

    return mask


def batch_inference_parallel(
    windows: Sequence[WindowSpec],
    articles_char_ids: Sequence[Sequence[int]],
    model: torch.nn.Module,
    codec,  # CharCodec instance with cls_id, sep_id
    device: torch.device,
    batch_size: int = 16,
    left_base: float = 0.1,
    right_base: float = 0.5,
) -> Tuple[List[np.ndarray], int]:
    """Run model.forward on every window in `windows`, applying asymmetric_mask
    to each. Returns (labels_per_article, n_forward_calls). labels_per_article
    is a list of int arrays, one per article, where labels[i] is the LAST
    window's prediction at position i (merge.last_window_wins) — the caller
    can override this with merge.center_priority or any other rule.

    articles_char_ids is indexed by window.article_idx. All windows are sorted
    by length and batched together across articles to minimize padding waste."""
    model.eval()
    n_articles = len(articles_char_ids)
    out_labels = [np.zeros(len(articles_char_ids[i]), dtype=np.int64) for i in range(n_articles)]

    # sort windows by content length (end - start) to reduce padding in each batch
    sorted_windows = sorted(windows, key=lambda w: w.end - w.start)

    n_calls = 0
    with torch.no_grad():
        for batch_start in range(0, len(sorted_windows), batch_size):
            batch_windows = sorted_windows[batch_start : batch_start + batch_size]
            batch_input_ids = []
            batch_masks = []
            for w in batch_windows:
                char_slice = articles_char_ids[w.article_idx][w.start : w.end]
                ids = [codec.cls_id] + list(char_slice) + [codec.sep_id]
                batch_input_ids.append(ids)
                mask = asymmetric_mask(
                    len(ids), w.responsible_start - w.start, w.responsible_end - w.start, left_base, right_base
                )
                batch_masks.append(mask)

            max_len = max(len(ids) for ids in batch_input_ids)
            padded_ids = []
            padded_masks = []
            for ids, mask in zip(batch_input_ids, batch_masks):
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [codec.pad_id] * pad_len)
                padded_masks.append(torch.cat([mask, torch.zeros(pad_len)]))

            input_ids_t = torch.tensor(padded_ids, dtype=torch.long, device=device)
            attention_mask_t = torch.stack(padded_masks).to(device)
            logits = model(input_ids_t, attention_mask=attention_mask_t)  # [batch, seq, 8]
            preds = logits.argmax(dim=-1).cpu().numpy()  # [batch, seq]
            n_calls += 1

            for i, w in enumerate(batch_windows):
                # extract the label predictions for this window's responsibility zone only
                # (positions 1..len(char_slice) in the padded sequence, skipping [CLS] at 0)
                window_len = w.end - w.start
                resp_offset_in_window = w.responsible_start - w.start
                resp_len = w.responsible_end - w.responsible_start
                pred_slice = preds[i, 1 + resp_offset_in_window : 1 + resp_offset_in_window + resp_len]
                out_labels[w.article_idx][w.responsible_start : w.responsible_end] = pred_slice

    return out_labels, n_calls


def apply_short_tail_windows(
    windows: List[WindowSpec], articles_char_ids: Sequence[Sequence[int]], tail_margin: int = 50
) -> List[WindowSpec]:
    """完善方案.md §4.3 核心改进2: if a window's right edge (w.end) would land
    within `tail_margin` chars of the article's true end, extend w.end to
    cover the entire tail so the model sees the real </s> boundary, not a
    fake mid-sentence cutoff. The window's responsibility zone is NOT changed
    (still ends at the originally planned responsible_end), because extending
    the zone would conflict with an adjacent window's zone. Returns a NEW list
    of WindowSpec (does not mutate the input)."""
    out = []
    for w in windows:
        article_len = len(articles_char_ids[w.article_idx])
        tail_start = article_len - tail_margin
        if w.end < article_len and w.end >= tail_start:
            # this window's right edge is in the tail margin; extend it to the end
            out.append(
                WindowSpec(
                    article_idx=w.article_idx,
                    start=w.start,
                    end=article_len,
                    responsible_start=w.responsible_start,
                    responsible_end=w.responsible_end,
                )
            )
        else:
            out.append(w)
    return out
