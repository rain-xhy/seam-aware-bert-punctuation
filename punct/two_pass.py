"""Two-pass seam-aware decoding.

Pass 1: ordinary parallel decoding (engine.decode_parallel) returning merged
class probabilities.
Selection: positions that sit close to an internal window edge of pass 1 and
whose top-1 probability is below `conf_thr` (or, for the random control, the
same number of positions drawn uniformly from the same article).
Pass 2: one longer window per cluster of selected positions, the window start
anchored just after a predicted sentence-final mark when one lies within
`anchor_range` characters to the left. Each selected position is re-labelled
by the pass-2 window in which it sits far from both edges.

All pass-2 windows of all articles are pooled into the same batched forward,
so the second pass costs extra forward tokens (reported as TER) but does not
add per-article round trips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from punct.engine import LogitsFn, PooledWindow, _run_batch, decode_parallel, pool_windows

SENT_FINAL = (2, 3, 4)


@dataclass
class PatchWindow:
    article: int
    start: int
    end: int
    positions: List[int]  # article-local positions this window re-labels


def seam_candidates(windows: Sequence[PooledWindow], lengths: Sequence[int], margin: int) -> Dict[int, np.ndarray]:
    """Per article: boolean mask of positions within `margin` chars of an
    INTERNAL edge of some pass-1 window (article start/end are not seams)."""
    masks = {a: np.zeros(n, dtype=bool) for a, n in enumerate(lengths)}
    for w in windows:
        n = lengths[w.article]
        if w.start > 0:
            masks[w.article][w.start:min(w.start + margin, n)] = True
        if w.end < n:
            masks[w.article][max(w.end - margin, 0):w.end] = True
    return masks


def select_positions(probs: Sequence[np.ndarray], masks: Dict[int, np.ndarray], conf_thr: float,
                     mode: str = "confidence", rng: Optional[np.random.Generator] = None,
                     budget: Optional[int] = None) -> Dict[int, np.ndarray]:
    """mode 'confidence': seam positions with top-1 prob < conf_thr.
    mode 'random': `budget` positions drawn uniformly over all articles'
    positions (same count as a confidence run, for the equal-budget control)."""
    if mode == "confidence":
        out = {}
        for a, p in enumerate(probs):
            low = (p.max(axis=1) < conf_thr) & masks[a]
            idx = np.flatnonzero(low)
            if idx.size:
                out[a] = idx
        return out
    if mode == "random":
        if rng is None or budget is None:
            raise ValueError("random mode needs rng and budget")
        lengths = np.array([len(p) for p in probs])
        total = int(lengths.sum())
        if total == 0 or budget <= 0:
            return {}
        flat = np.sort(rng.choice(total, size=min(budget, total), replace=False))
        offs = np.concatenate([[0], np.cumsum(lengths)])
        out: Dict[int, np.ndarray] = {}
        art = np.searchsorted(offs, flat, side="right") - 1
        for a in np.unique(art):
            out[int(a)] = (flat[art == a] - offs[a]).astype(np.int64)
        return out
    raise ValueError(f"unknown mode {mode!r}")


def _anchor_start(start: int, labels: np.ndarray, anchor_range: int) -> int:
    """Move `start` left to the position right after the nearest predicted
    sentence-final mark within anchor_range chars to its left."""
    if start <= 0 or anchor_range <= 0:
        return start
    lo = max(0, start - anchor_range)
    seg = labels[lo:start]
    hits = np.flatnonzero(np.isin(seg, SENT_FINAL))
    return lo + int(hits[-1]) + 1 if hits.size else start


def plan_patches(positions: Dict[int, np.ndarray], lengths: Sequence[int], labels: Sequence[np.ndarray],
                 patch_window: int = 400, edge_margin: int = 128, anchor_range: int = 16) -> List[PatchWindow]:
    """Cluster each article's selected positions into windows. A window
    re-labels only positions at least `edge_margin` chars from its edges,
    except at the article's true start/end."""
    if patch_window <= 2 * edge_margin:
        raise ValueError("patch_window must exceed 2*edge_margin")
    out: List[PatchWindow] = []
    for a, pos in positions.items():
        n = lengths[a]
        pos = np.sort(pos)
        i = 0
        while i < len(pos):
            q = int(pos[i])
            if n <= patch_window:
                s, e = 0, n
            else:
                s = max(0, min(q - edge_margin, n - patch_window))
                s = _anchor_start(s, labels[a], anchor_range) if s > 0 else 0
                e = min(s + patch_window, n)
            lo = s if s == 0 else s + edge_margin
            hi = e if e == n else e - edge_margin
            take = [int(p) for p in pos[i:] if lo <= p < hi]
            if not take:  # anchoring moved the window; q itself must still be covered
                take = [q]
                if not (s <= q < e):
                    raise AssertionError("patch window does not contain its position")
            out.append(PatchWindow(a, s, e, take))
            i += len(take)
    return out


def run_patches(articles: Sequence[Sequence[int]], patches: Sequence[PatchWindow], logits_fn: LogitsFn,
                special: Tuple[int, int, int], device: torch.device, batch_size: int = 8,
                num_labels: int = 8) -> Tuple[List[Tuple[int, int, int]], int, int]:
    """-> ([(article, pos, new_label)], forward_tokens, n_oom_retries)."""
    cls_id, sep_id, pad_id = special
    order = sorted(range(len(patches)), key=lambda k: patches[k].end - patches[k].start)
    results: List[Tuple[int, int, int]] = []
    tokens = sum(p.end - p.start for p in patches)
    bs, i, retries = batch_size, 0, 0
    while i < len(order):
        idx = order[i:i + bs]
        chunks = [articles[patches[k].article][patches[k].start:patches[k].end] for k in idx]
        try:
            probs = _run_batch(logits_fn, chunks, cls_id, sep_id, pad_id, device)
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs, retries = max(bs // 2, 1), retries + 1
            continue
        pred = probs.argmax(-1).cpu().numpy()
        for row, k in enumerate(idx):
            p = patches[k]
            for pos in p.positions:
                results.append((p.article, pos, int(pred[row, 1 + pos - p.start])))
        i += len(idx)
    return results, tokens, retries


def decode_two_pass(
    articles: Sequence[Sequence[int]], logits_fn: LogitsFn, special: Tuple[int, int, int], device: torch.device,
    window: int = 400, overlap: int = 100, tail: str = "short", mask_left: int = 0, mask_right: int = 0,
    profile: str = "hard", batch_size: int = 8, seam_margin: int = 16, conf_thr: float = 0.9,
    patch_window: int = 400, edge_margin: int = 128, anchor_range: int = 16, select: str = "confidence",
    rng: Optional[np.random.Generator] = None, budget: Optional[int] = None, patch: bool = True,
):
    """-> (labels per article, info dict). `select='random'` draws `budget`
    random positions (the equal-budget control); `patch=False` returns pass 1."""
    cls_id, sep_id, pad_id = special
    lengths = [len(a) for a in articles]
    labels, stats, probs = decode_parallel(
        articles, logits_fn, cls_id, sep_id, pad_id, device, window=window, overlap=overlap, tail=tail,
        mask_left=mask_left, mask_right=mask_right, profile=profile, batch_size=batch_size, return_probs=True)
    info = {"pass1_ter": stats.ter, "pass1_windows": stats.n_windows, "pass1_tokens": stats.forward_tokens,
            "n_selected": 0, "n_patch_windows": 0, "patch_tokens": 0, "n_changed": 0, "ter": stats.ter,
            "pass1_seconds": stats.total_seconds}
    if not patch:
        return labels, info
    pooled = pool_windows(lengths, window, overlap, tail)
    masks = seam_candidates(pooled, lengths, seam_margin)
    sel = select_positions(probs, masks, conf_thr, select, rng, budget)
    info["n_selected"] = int(sum(len(v) for v in sel.values()))
    patches = plan_patches(sel, lengths, labels, patch_window, edge_margin, anchor_range)
    info["n_patch_windows"] = len(patches)
    res, tokens, retries = run_patches(articles, patches, logits_fn, special, device, batch_size)
    info["patch_tokens"], info["n_oom_retries"] = tokens, retries
    out = [l.copy() for l in labels]
    for a, pos, lab in res:
        if out[a][pos] != lab:
            info["n_changed"] += 1
        out[a][pos] = lab
    total = max(sum(lengths), 1)
    info["ter"] = (stats.forward_tokens + tokens) / total
    return out, info
