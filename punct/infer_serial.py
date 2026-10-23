"""Serial decoding baselines: the teacher's own method, Tang 2021's two
serial variants, and a cross-article batched version of the teacher's
method (完善方案.md §4.1).

All four share the defining property that makes them serial: the NEXT
window's start position is read off of the model's own prediction on the
CURRENT window, so a single article's windows cannot all be planned ahead
of time the way window_plan.plan_windows() does for the parallel methods.
This is exactly the dependency 完善方案.md's "核心不变" section calls out.

teacher_serial         - read up to `window` chars, find the last position
                          among the first `commit_prefix` predictions whose
                          label is a sentence-final mark (完善方案.md §4.1:
                          "在前 300 字附近最后一个句末标点处截断"), commit up
                          to and including it, continue from there. No
                          sentence-final mark found -> commit `commit_prefix`
                          chars outright.
tang_serial_v1          - 俞敬松 2019's decoding rule: read `window` chars,
                          keep only the first 1-2 sentences of the output,
                          discard the rest, continue from there. Reproduces
                          the DECODING rule only (not the training length,
                          per 完善方案.md §4.1's explicit caveat).
tang_serial_v2          - Tang 2021's own improvement: read `window` chars,
                          drop only the LAST sentence, keep everything
                          before it, continue from there.
cross_article_serial    - the teacher's own method, but instead of finishing
                           one article before starting the next, every
                           article's current window is batched together at
                           each step (完善方案.md §4.1, "跨文章串行" row) —
                           this breaks the single-article serial dependency
                           without changing the decoding rule itself.

Every function returns (labels, n_forward_calls) so infer_serial and
infer_parallel can be compared on "how many model calls did this take" as
well as on wall-clock time (完善方案.md §6, TER / forward-call counts).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

ModelFn = Callable[[Sequence[int]], np.ndarray]  # char ids -> predicted label ids, same length


def _last_sentence_final_index(labels: Sequence[int], sentence_final_labels, search_within: int) -> int:
    """Index (0-based, within the first `search_within` labels) of the last
    sentence-final predicted label, or -1 if none is found."""
    final_set = set(sentence_final_labels)
    last = -1
    for i, lab in enumerate(labels[:search_within]):
        if lab in final_set:
            last = i
    return last


def teacher_serial(
    char_ids: Sequence[int], model_fn: ModelFn, window: int = 400, commit_prefix: int = 300,
    sentence_final_labels=(2, 3, 4),
) -> Tuple[List[int], int]:
    """完善方案.md §4.1 "老师 400/300 串行": window=400, commit_prefix=300."""
    n = len(char_ids)
    labels = [0] * n
    pos = 0
    calls = 0
    while pos < n:
        chunk = char_ids[pos:pos + window]
        chunk_labels = model_fn(chunk)
        calls += 1
        if pos + len(chunk) >= n:
            # last window: commit everything, no more text to gain context from
            for i, lab in enumerate(chunk_labels):
                labels[pos + i] = lab
            pos += len(chunk)
            continue
        cut = _last_sentence_final_index(chunk_labels, sentence_final_labels, min(commit_prefix, len(chunk)))
        commit_len = (cut + 1) if cut >= 0 else min(commit_prefix, len(chunk))
        for i in range(commit_len):
            labels[pos + i] = chunk_labels[i]
        pos += commit_len
    return labels, calls


def tang_serial_v1(
    char_ids: Sequence[int], model_fn: ModelFn, window: int = 64, keep_sentences: int = 2,
    sentence_final_labels=(2, 3, 4),
) -> Tuple[List[int], int]:
    """俞敬松 2019's decoding rule, reproduced per 完善方案.md §4.1: window=64,
    keep only the first `keep_sentences` sentences of each window's output."""
    n = len(char_ids)
    labels = [0] * n
    pos = 0
    calls = 0
    while pos < n:
        chunk = char_ids[pos:pos + window]
        chunk_labels = model_fn(chunk)
        calls += 1
        if pos + len(chunk) >= n:
            for i, lab in enumerate(chunk_labels):
                labels[pos + i] = lab
            pos += len(chunk)
            continue
        final_set = set(sentence_final_labels)
        cut_positions = [i for i, lab in enumerate(chunk_labels) if lab in final_set]
        if len(cut_positions) >= keep_sentences:
            commit_len = cut_positions[keep_sentences - 1] + 1
        elif cut_positions:
            commit_len = cut_positions[-1] + 1
        else:
            commit_len = len(chunk)  # no sentence boundary found at all: commit the whole window
        for i in range(commit_len):
            labels[pos + i] = chunk_labels[i]
        pos += commit_len
    return labels, calls


def tang_serial_v2(
    char_ids: Sequence[int], model_fn: ModelFn, window: int = 125, sentence_final_labels=(2, 3, 4),
) -> Tuple[List[int], int]:
    """Tang 2021's own improved serial rule, per 完善方案.md §4.1: window=125,
    drop only the LAST predicted sentence, keep everything before it."""
    n = len(char_ids)
    labels = [0] * n
    pos = 0
    calls = 0
    while pos < n:
        chunk = char_ids[pos:pos + window]
        chunk_labels = model_fn(chunk)
        calls += 1
        if pos + len(chunk) >= n:
            for i, lab in enumerate(chunk_labels):
                labels[pos + i] = lab
            pos += len(chunk)
            continue
        cut = _last_sentence_final_index(chunk_labels, sentence_final_labels, len(chunk))
        commit_len = (cut + 1) if cut >= 0 else len(chunk)
        for i in range(commit_len):
            labels[pos + i] = chunk_labels[i]
        pos += commit_len
    return labels, calls


@dataclass
class _ArticleCursor:
    char_ids: Sequence[int]
    pos: int = 0

    @property
    def done(self) -> bool:
        return self.pos >= len(self.char_ids)


def cross_article_serial(
    articles: Sequence[Sequence[int]], batch_model_fn: Callable[[List[Sequence[int]]], List[np.ndarray]],
    window: int = 400, commit_prefix: int = 300, sentence_final_labels=(2, 3, 4),
) -> Tuple[List[List[int]], int]:
    """完善方案.md §4.1 "跨文章串行": the teacher's own decoding rule, but at
    every step every still-unfinished article contributes its current
    window to one batch call, instead of finishing one article before
    starting the next. batch_model_fn takes a list of char-id chunks and
    returns one predicted-label array per chunk. Returns (labels_per_article,
    n_batch_calls) — n_batch_calls counts BATCH calls, not per-article calls,
    which is the whole point of this baseline."""
    cursors = [_ArticleCursor(a) for a in articles]
    out_labels: List[List[int]] = [[0] * len(a) for a in articles]
    calls = 0
    while any(not c.done for c in cursors):
        active_idx = [i for i, c in enumerate(cursors) if not c.done]
        chunks = [cursors[i].char_ids[cursors[i].pos:cursors[i].pos + window] for i in active_idx]
        chunk_labels_list = batch_model_fn(chunks)
        calls += 1
        for list_pos, art_idx in enumerate(active_idx):
            cursor = cursors[art_idx]
            chunk_labels = chunk_labels_list[list_pos]
            chunk_len = len(chunks[list_pos])
            is_last = cursor.pos + chunk_len >= len(cursor.char_ids)
            if is_last:
                commit_len = chunk_len
            else:
                cut = _last_sentence_final_index(chunk_labels, sentence_final_labels, min(commit_prefix, chunk_len))
                commit_len = (cut + 1) if cut >= 0 else min(commit_prefix, chunk_len)
            for i in range(commit_len):
                out_labels[art_idx][cursor.pos + i] = chunk_labels[i]
            cursor.pos += commit_len
    return out_labels, calls
