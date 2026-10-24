"""Batched window decoding engine shared by every parallel method.

Pipeline: plan windows (window_plan) -> sort the pooled windows of ALL
articles by length -> batched forward (padding + attention mask, OOM
halve-and-retry) -> softmax -> GPU-side weighted accumulation with
index_add_ -> argmax -> one transfer back to the CPU.

The forward function is injected (`logits_fn(input_ids, attention_mask) ->
logits`), so the same code is tested with a position-determined fake model
and run with the real BERT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from punct.window_plan import Window, plan_windows

LogitsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def load_finetuned(bert_dir: str, ckpt_path: str, device: torch.device, num_labels: int = 8):
    """bert-base-chinese structure + state_dict written by train.py."""
    from transformers import BertForTokenClassification

    model = BertForTokenClassification.from_pretrained(bert_dir, num_labels=num_labels, local_files_only=True)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()


def make_logits_fn(model, use_bf16: bool = True) -> LogitsFn:
    def fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if use_bf16 and input_ids.is_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            return model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    return fn


@dataclass
class DecodeStats:
    n_windows: int = 0
    forward_tokens: int = 0       # chars fed to the model (excl. CLS/SEP)
    text_chars: int = 0
    n_forward_calls: int = 0
    n_oom_retries: int = 0
    forward_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_mem_mb: float = 0.0
    batch_size_used: int = 0

    @property
    def ter(self) -> float:
        return self.forward_tokens / max(self.text_chars, 1)


@dataclass
class PooledWindow:
    article: int
    start: int
    end: int


def pool_windows(lengths: Sequence[int], window: int, overlap: int, tail: str) -> List[PooledWindow]:
    out: List[PooledWindow] = []
    for a, n in enumerate(lengths):
        for w in plan_windows(n, window, overlap, tail):
            out.append(PooledWindow(a, w.start, w.end))
    return out


def edge_weights(length: int, at_article_start: bool, at_article_end: bool, mask_left: int, mask_right: int,
                 profile: str = "hard", edep=None, edep_floor: float = 0.1) -> np.ndarray:
    """Per-position merge weight for one window. 'hard': 1 inside, 0 within
    mask_left/mask_right of an internal edge (article start/end exempt).
    'gauss': Gaussian centred on the window, never exactly zero, edges at an
    article boundary get the centre weight."""
    if profile == "hard":
        w = np.ones(length, dtype=np.float32)
        if not at_article_start and mask_left > 0:
            w[:mask_left] = 0.0
        if not at_article_end and mask_right > 0:
            w[max(length - mask_right, 0):] = 0.0
        return w
    if profile == "gauss":
        idx = np.arange(length, dtype=np.float32)
        centre = (length - 1) / 2.0
        sigma = max(0.25 * length, 1e-6)
        w = np.exp(-0.5 * ((idx - centre) / sigma) ** 2).astype(np.float32) + 1e-6
        if at_article_start:
            w[: int(centre) + 1] = np.maximum(w[: int(centre) + 1], w[int(centre)])
        if at_article_end:
            w[int(centre):] = np.maximum(w[int(centre):], w[int(centre)])
        return w
    if profile == "edep":
        # empirical weight: 1 - error rate at this distance from the window
        # edge (EDEP-L from the left, EDEP-R from the right), the worse of the
        # two sides, floored so no position is fully masked. Edges that sit on
        # the article's true start/end are not seams and keep weight 1.
        if edep is None:
            raise ValueError("profile='edep' needs edep=(left_rates, right_rates)")
        left_err, right_err = (np.asarray(e, dtype=np.float32) for e in edep)
        err = np.zeros(length, dtype=np.float32)
        d = np.arange(length)
        if not at_article_start:
            inside = d < len(left_err)
            err[inside] = np.maximum(err[inside], left_err[d[inside]])
        if not at_article_end:
            dr = length - 1 - d
            inside = dr < len(right_err)
            err[inside] = np.maximum(err[inside], right_err[dr[inside]])
        return np.maximum(1.0 - err, edep_floor).astype(np.float32)
    raise ValueError(f"unknown profile {profile!r}")


def _run_batch(logits_fn: LogitsFn, chunks: List[Sequence[int]], cls_id: int, sep_id: int, pad_id: int,
               device: torch.device) -> torch.Tensor:
    """-> probs [B, Lmax, C] (positions 1..len of each row are the chars)."""
    max_len = max(len(c) for c in chunks) + 2
    ids = np.full((len(chunks), max_len), pad_id, dtype=np.int64)
    mask = np.zeros((len(chunks), max_len), dtype=np.int64)
    for i, c in enumerate(chunks):
        n = len(c)
        ids[i, 0] = cls_id
        ids[i, 1:1 + n] = c
        ids[i, 1 + n] = sep_id
        mask[i, : n + 2] = 1
    ids_t = torch.from_numpy(ids).to(device, non_blocking=True)
    mask_t = torch.from_numpy(mask).to(device, non_blocking=True)
    return torch.softmax(logits_fn(ids_t, mask_t), dim=-1)


def decode_parallel(
    articles: Sequence[Sequence[int]], logits_fn: LogitsFn, cls_id: int, sep_id: int, pad_id: int,
    device: torch.device, window: int, overlap: int, tail: str = "short",
    mask_left: int = 0, mask_right: int = 0, profile: str = "hard", batch_size: int = 16,
    num_labels: int = 8, return_probs: bool = False, windows: Optional[List[PooledWindow]] = None,
    edep=None,
):
    """Parallel sliding-window decoding of many articles at once.
    Returns (labels per article, DecodeStats[, probs per article])."""
    t_all = time.perf_counter()
    lengths = [len(a) for a in articles]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total = int(offsets[-1])
    pooled = windows if windows is not None else pool_windows(lengths, window, overlap, tail)
    pooled = sorted(pooled, key=lambda w: w.end - w.start)

    stats = DecodeStats(n_windows=len(pooled), text_chars=total,
                        forward_tokens=sum(w.end - w.start for w in pooled))
    prob_sum = torch.zeros(total, num_labels, dtype=torch.float32, device=device)
    w_sum = torch.zeros(total, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bs = batch_size
    i = 0
    while i < len(pooled):
        batch = pooled[i:i + bs]
        chunks = [articles[w.article][w.start:w.end] for w in batch]
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = _run_batch(logits_fn, chunks, cls_id, sep_id, pad_id, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats.forward_seconds += time.perf_counter() - t0
        except torch.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(bs // 2, 1)
            stats.n_oom_retries += 1
            continue
        stats.n_forward_calls += 1
        lens = [w.end - w.start for w in batch]
        # one scatter per batch: gather valid rows, build global indices + weights
        rows, idxs, wts = [], [], []
        for b, w in enumerate(batch):
            n = lens[b]
            rows.append(probs[b, 1:1 + n])
            base = int(offsets[w.article]) + w.start
            idxs.append(torch.arange(base, base + n, device=device))
            wv = edge_weights(n, w.start == 0, w.end == lengths[w.article], mask_left, mask_right, profile, edep)
            wts.append(torch.from_numpy(wv).to(device))
        rows_t = torch.cat(rows)
        idx_t = torch.cat(idxs)
        w_t = torch.cat(wts)
        prob_sum.index_add_(0, idx_t, rows_t * w_t[:, None])
        w_sum.index_add_(0, idx_t, w_t)
        i += len(batch)

    if total and bool((w_sum <= 0).any().item()):
        raise ValueError("some positions received zero merge weight; mask_left+mask_right too large for the overlap")
    merged = prob_sum / w_sum.clamp_min(1e-12)[:, None]
    labels = merged.argmax(dim=-1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
        stats.peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20
    stats.total_seconds = time.perf_counter() - t_all
    stats.batch_size_used = bs
    per_article = [labels[offsets[a]:offsets[a + 1]] for a in range(len(articles))]
    if return_probs:
        pm = merged.cpu().numpy()
        return per_article, stats, [pm[offsets[a]:offsets[a + 1]] for a in range(len(articles))]
    return per_article, stats
