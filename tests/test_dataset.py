"""Tests for punct.dataset — pure Python/numpy, no torch or GPU needed."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from punct.dataset import IGNORE_INDEX, CropConfig, collate_batch, pick_crop


def make_article(length: int) -> np.ndarray:
    return np.arange(length, dtype=np.int32)


class TestPickCrop(unittest.TestCase):
    def test_too_short_article_returns_none(self):
        cfg = CropConfig(min_len=32, max_len=510)
        rng = random.Random(0)
        article = make_article(10)
        result = pick_crop(article, article, 0, cfg, rng)
        self.assertIsNone(result)

    def test_crop_length_within_bounds(self):
        cfg = CropConfig(min_len=32, max_len=510, edge_aware_prob=0.0)
        rng = random.Random(0)
        article = make_article(1000)
        for _ in range(200):
            result = pick_crop(article, article, 0, cfg, rng)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(len(result.char_ids), cfg.min_len)
            self.assertLessEqual(len(result.char_ids), cfg.max_len)

    def test_crop_never_exceeds_article_length(self):
        cfg = CropConfig(min_len=32, max_len=510, edge_aware_prob=0.0)
        rng = random.Random(0)
        article = make_article(60)  # shorter than max_len
        for _ in range(200):
            result = pick_crop(article, article, 0, cfg, rng)
            self.assertLessEqual(result.end, 60)
            self.assertLessEqual(len(result.char_ids), 60)

    def test_crop_characters_match_article_slice(self):
        cfg = CropConfig(min_len=32, max_len=510, edge_aware_prob=0.0)
        rng = random.Random(0)
        article = make_article(1000)
        result = pick_crop(article, article, 0, cfg, rng)
        np.testing.assert_array_equal(result.char_ids, article[result.start:result.end])

    def test_edge_aware_prob_zero_means_full_length_end(self):
        # with edge_aware_prob=0, 'end' should equal start + crop_len exactly
        # (never further shortened) — checked indirectly via reproducibility
        cfg = CropConfig(min_len=100, max_len=100, edge_aware_prob=0.0)
        rng = random.Random(0)
        article = make_article(1000)
        result = pick_crop(article, article, 0, cfg, rng)
        self.assertEqual(result.end - result.start, 100)

    def test_edge_aware_prob_one_can_shorten_right_edge(self):
        cfg = CropConfig(min_len=100, max_len=100, edge_aware_prob=1.0, edge_aware_min_right=1)
        rng = random.Random(0)
        article = make_article(1000)
        shortened_seen = False
        for _ in range(100):
            result = pick_crop(article, article, 0, cfg, rng)
            if result.end - result.start < 100:
                shortened_seen = True
                break
        self.assertTrue(shortened_seen, "edge_aware_prob=1.0 should shorten at least one crop in 100 tries")

    def test_center_only_supervision_masks_both_edges(self):
        cfg = CropConfig(min_len=200, max_len=200, edge_aware_prob=0.0,
                          center_only_supervision=True, center_margin=50)
        rng = random.Random(0)
        article = make_article(1000)
        result = pick_crop(article, article, 0, cfg, rng)
        self.assertEqual(len(result.label_ids), 200)
        self.assertTrue(all(l == IGNORE_INDEX for l in result.label_ids[:50]))
        self.assertTrue(all(l == IGNORE_INDEX for l in result.label_ids[-50:]))
        self.assertTrue(any(l != IGNORE_INDEX for l in result.label_ids[50:-50]))

    def test_center_only_supervision_short_crop_masks_nothing(self):
        # crop shorter than 2*margin: masking the whole thing would leave
        # zero supervised positions, so the function should leave it alone
        cfg = CropConfig(min_len=10, max_len=10, edge_aware_prob=0.0,
                          center_only_supervision=True, center_margin=50)
        rng = random.Random(0)
        article = make_article(1000)
        result = pick_crop(article, article, 0, cfg, rng)
        self.assertTrue(all(l != IGNORE_INDEX for l in result.label_ids))

    def test_article_index_is_passed_through(self):
        cfg = CropConfig(min_len=32, max_len=510)
        rng = random.Random(0)
        article = make_article(1000)
        result = pick_crop(article, article, 42, cfg, rng)
        self.assertEqual(result.article_index, 42)

    def test_reproducible_with_same_seed(self):
        cfg = CropConfig(min_len=32, max_len=510)
        article = make_article(1000)
        r1 = pick_crop(article, article, 0, cfg, random.Random(123))
        r2 = pick_crop(article, article, 0, cfg, random.Random(123))
        self.assertEqual(r1.start, r2.start)
        self.assertEqual(r1.end, r2.end)


class TestCollateBatch(unittest.TestCase):
    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            collate_batch([], cls_id=101, sep_id=102, pad_id=0)

    def test_single_crop_shape(self):
        cfg = CropConfig(min_len=10, max_len=10, edge_aware_prob=0.0)
        article = make_article(100)
        crop = pick_crop(article, article, 0, cfg, random.Random(0))
        input_ids, mask, labels = collate_batch([crop], cls_id=101, sep_id=102, pad_id=0)
        self.assertEqual(input_ids.shape, (1, 12))  # 10 chars + CLS + SEP
        self.assertEqual(mask.shape, (1, 12))
        self.assertEqual(labels.shape, (1, 12))

    def test_cls_sep_and_padding_get_ignore_index(self):
        cfg = CropConfig(min_len=10, max_len=10, edge_aware_prob=0.0)
        article = make_article(100)
        c1 = pick_crop(article, article, 0, cfg, random.Random(0))
        c1 = type(c1)(char_ids=c1.char_ids, label_ids=[1] * len(c1.char_ids),
                       article_index=0, start=c1.start, end=c1.end)
        input_ids, mask, labels = collate_batch([c1], cls_id=101, sep_id=102, pad_id=0)
        self.assertEqual(labels[0, 0], IGNORE_INDEX)  # CLS position
        self.assertEqual(labels[0, -1], IGNORE_INDEX)  # SEP position

    def test_different_lengths_padded_to_batch_max(self):
        cfg_short = CropConfig(min_len=5, max_len=5, edge_aware_prob=0.0)
        cfg_long = CropConfig(min_len=20, max_len=20, edge_aware_prob=0.0)
        article = make_article(100)
        short_crop = pick_crop(article, article, 0, cfg_short, random.Random(0))
        long_crop = pick_crop(article, article, 0, cfg_long, random.Random(0))
        input_ids, mask, labels = collate_batch([short_crop, long_crop], cls_id=101, sep_id=102, pad_id=0)
        self.assertEqual(input_ids.shape[1], 22)  # 20 + CLS + SEP
        # short row: mask should be 1 for CLS+5 chars+SEP = 7, then 0 (padding)
        self.assertEqual(mask[0].sum(), 7)
        self.assertEqual(mask[1].sum(), 22)

    def test_cls_id_at_position_zero(self):
        cfg = CropConfig(min_len=10, max_len=10, edge_aware_prob=0.0)
        article = make_article(100)
        crop = pick_crop(article, article, 0, cfg, random.Random(0))
        input_ids, _, _ = collate_batch([crop], cls_id=101, sep_id=102, pad_id=0)
        self.assertEqual(input_ids[0, 0], 101)

    def test_sep_id_right_after_content(self):
        cfg = CropConfig(min_len=10, max_len=10, edge_aware_prob=0.0)
        article = make_article(100)
        crop = pick_crop(article, article, 0, cfg, random.Random(0))
        input_ids, _, _ = collate_batch([crop], cls_id=101, sep_id=102, pad_id=0)
        self.assertEqual(input_ids[0, 11], 102)  # position 1 + 10 chars = 11


if __name__ == "__main__":
    unittest.main()


class TestShortArticleFallback(unittest.TestCase):
    def test_fixed_length_skips_short_articles(self):
        import tempfile, os
        import numpy as np
        from punct.dataset import CropConfig, WindowDataset
        with tempfile.TemporaryDirectory() as d:
            chars = np.empty(4, dtype=object)
            labs = np.empty(4, dtype=object)
            for i, n in enumerate((40, 50, 60, 300)):
                chars[i] = np.arange(n, dtype=np.int32) + 200
                labs[i] = np.zeros(n, dtype=np.int32)
            np.save(os.path.join(d, "train_articles_char.npy"), chars, allow_pickle=True)
            np.save(os.path.join(d, "train_articles_label.npy"), labs, allow_pickle=True)
            ds = WindowDataset(d, "train", CropConfig(min_len=256, max_len=256, edge_aware_prob=0.0))
            for idx in range(len(ds)):
                self.assertEqual(len(ds[idx].char_ids), 256)
