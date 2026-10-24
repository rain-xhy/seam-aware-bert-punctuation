"""Training sample generation: variable-length random crops with edge-awareness.

完善方案.md §3.2: a training window is not a fixed-length slice of a fixed
article boundary. Instead, for every sample we pick a random article, a
random length L in [min_len, max_len], and a random start offset, so the
model sees windows that start and end mid-sentence — matching what the
parallel inference windows will actually look like.

Edge-aware cropping additionally shortens the RIGHT end of some samples by a
further random amount, so the model also trains on "right context is almost
gone" positions (完善方案.md §3.2, the asymmetric-masking rationale: the
label is "what punctuation follows this character", which depends mostly on
the right-hand context).

This module has no torch dependency beyond tensor construction, so its
core sampling logic (what offsets get picked, where -100 goes) is unit-
tested without needing a GPU or a model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

IGNORE_INDEX = -100


@dataclass
class CropConfig:
    min_len: int = 32
    max_len: int = 510
    edge_aware_prob: float = 0.3  # fraction of samples that get a shortened right edge
    edge_aware_min_right: int = 1  # shortened right context can go down to this many chars
    center_only_supervision: bool = False  # True for the "only supervise centre" ablation (T4)
    center_margin: int = 50  # chars left unsupervised on each side when center_only_supervision


@dataclass
class CropResult:
    char_ids: List[int]
    label_ids: List[int]  # IGNORE_INDEX where not supervised
    article_index: int
    start: int
    end: int  # exclusive, into the article's own char stream (after any right-edge shortening)


def pick_crop(
    article_char_ids: np.ndarray, article_label_ids: np.ndarray, article_index: int,
    cfg: CropConfig, rng: random.Random,
) -> Optional[CropResult]:
    """Draw one training crop from one article. Returns None if the article
    is shorter than cfg.min_len (too short to crop at all)."""
    article_len = len(article_char_ids)
    if article_len < cfg.min_len:
        return None

    max_len = min(cfg.max_len, article_len)
    crop_len = rng.randint(cfg.min_len, max_len)
    start = rng.randint(0, article_len - crop_len)
    end = start + crop_len

    if rng.random() < cfg.edge_aware_prob and crop_len > cfg.edge_aware_min_right:
        shortened_len = rng.randint(cfg.edge_aware_min_right, crop_len)
        end = start + shortened_len

    char_ids = article_char_ids[start:end].tolist()
    label_ids = article_label_ids[start:end].tolist()

    if cfg.center_only_supervision:
        label_ids = _apply_center_only_mask(label_ids, cfg.center_margin)

    return CropResult(char_ids=char_ids, label_ids=label_ids, article_index=article_index, start=start, end=end)


def _apply_center_only_mask(label_ids: List[int], margin: int) -> List[int]:
    """Set IGNORE_INDEX on the first/last `margin` positions, keep the rest.
    If the crop is too short to have a non-empty centre, nothing is masked
    (supervising the whole short crop is better than supervising nothing)."""
    n = len(label_ids)
    if n <= 2 * margin:
        return label_ids
    masked = list(label_ids)
    for i in range(margin):
        masked[i] = IGNORE_INDEX
        masked[n - 1 - i] = IGNORE_INDEX
    return masked


def collate_batch(crops: List[CropResult], cls_id: int, sep_id: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a list of crops to the batch's max length, adding [CLS]/[SEP] and
    attention mask. Returns (input_ids, attention_mask, label_ids), all
    shape (batch, max_len + 2). [CLS], [SEP] and padding all get IGNORE_INDEX."""
    if not crops:
        raise ValueError("collate_batch called with an empty list")
    max_len = max(len(c.char_ids) for c in crops)
    batch = len(crops)
    seq_len = max_len + 2

    input_ids = np.full((batch, seq_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((batch, seq_len), dtype=np.int64)
    label_ids = np.full((batch, seq_len), IGNORE_INDEX, dtype=np.int64)

    for row, crop in enumerate(crops):
        n = len(crop.char_ids)
        input_ids[row, 0] = cls_id
        input_ids[row, 1:1 + n] = crop.char_ids
        input_ids[row, 1 + n] = sep_id
        attention_mask[row, :2 + n] = 1
        label_ids[row, 1:1 + n] = crop.label_ids

    return input_ids, attention_mask, label_ids


class WindowDataset:
    """PyTorch Dataset that loads articles from prepared_data/ and generates
    random crops on the fly."""
    def __init__(self, data_dir: str, split: str, cfg: CropConfig, seed: int = 42):
        import os
        self.articles_char = np.load(os.path.join(data_dir, f"{split}_articles_char.npy"), allow_pickle=True)
        self.articles_label = np.load(os.path.join(data_dir, f"{split}_articles_label.npy"), allow_pickle=True)
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.epoch_seed = seed

    def __len__(self):
        return len(self.articles_char) * 10  # 10 crops per article per epoch

    def __getitem__(self, idx):
        n = len(self.articles_char)
        for k in range(n):
            article_idx = (idx + k) % n
            crop = pick_crop(self.articles_char[article_idx], self.articles_label[article_idx],
                             article_idx, self.cfg, self.rng)
            if crop is not None:
                return crop
        raise ValueError(f"no article is at least min_len={self.cfg.min_len} characters long")

    def set_epoch(self, epoch: int):
        """Re-seed RNG for each epoch to get different crops."""
        self.rng = random.Random(self.epoch_seed + epoch)


def collate_windows(batch: List[CropResult], cls_id: int, sep_id: int, pad_id: int):
    """Wrapper for collate_batch that returns torch tensors."""
    import torch
    input_ids, attention_mask, label_ids = collate_batch(batch, cls_id, sep_id, pad_id)
    return {
        'input_ids': torch.from_numpy(input_ids),
        'attention_mask': torch.from_numpy(attention_mask),
        'labels': torch.from_numpy(label_ids)
    }
