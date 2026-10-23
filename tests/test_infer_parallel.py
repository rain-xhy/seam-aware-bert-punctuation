"""Tests for punct.infer_parallel: asymmetric masking, batch inference, and
short-tail-window logic. Uses a fake model (torch.nn.Module subclass that
returns deterministic position-based labels, same technique as test_merge.py)
so the LOGIC is tested independently of BERT quality."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from punct.infer_parallel import (
    WindowSpec,
    apply_short_tail_windows,
    asymmetric_mask,
    batch_inference_parallel,
)

NONE, COMMA, PERIOD, QUESTION, EXCLAIM = 0, 1, 2, 3, 4


class FakeCodec:
    """Mimics CharCodec's interface for testing."""

    cls_id = 101
    sep_id = 102
    pad_id = 0


class FakeModel(torch.nn.Module):
    """Returns logits whose argmax is PERIOD at every position divisible by 50,
    COMMA at every position divisible by 10 (but not 50), NONE elsewhere."""

    def forward(self, input_ids, attention_mask=None):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 8, dtype=torch.float32)
        for b in range(batch):
            for s in range(seq):
                char_id = input_ids[b, s].item()
                if char_id in (FakeCodec.cls_id, FakeCodec.sep_id, FakeCodec.pad_id):
                    logits[b, s, NONE] = 10.0  # special tokens always NONE
                elif char_id % 50 == 49:  # global ids 49, 99, 149, ... -> PERIOD
                    logits[b, s, PERIOD] = 10.0
                elif char_id % 10 == 9:  # 9, 19, 29, 39, 59, ... -> COMMA
                    logits[b, s, COMMA] = 10.0
                else:
                    logits[b, s, NONE] = 10.0
        return logits


class TestAsymmetricMask(unittest.TestCase):
    def test_responsibility_zone_never_masked(self):
        # seq_len=10 (indices 0..9), responsible for indices 3..6 (after +1 shift for [CLS], that's 4..7 in the mask)
        mask = asymmetric_mask(10, responsible_start=3, responsible_end=6)
        self.assertEqual(mask[4].item(), 1.0)
        self.assertEqual(mask[5].item(), 1.0)
        self.assertEqual(mask[6].item(), 1.0)

    def test_cls_and_sep_never_masked(self):
        mask = asymmetric_mask(10, 3, 6)
        self.assertEqual(mask[0].item(), 1.0)  # [CLS]
        self.assertEqual(mask[9].item(), 1.0)  # [SEP]

    def test_left_side_drops_quadratically(self):
        mask = asymmetric_mask(100, 50, 60, left_base=0.1)
        # positions 1..50 are left of the zone; the one immediately left of
        # the zone (index 50, which is responsible_start+1) should have the
        # SMALLEST quadratic penalty; the one furthest left (index 1) should
        # have the LARGEST penalty
        self.assertLess(mask[1].item(), mask[50].item())

    def test_right_side_constant_masking(self):
        mask = asymmetric_mask(100, 40, 50, right_base=0.5)
        # positions 51..98 are right of the zone (99 is [SEP], always 1.0)
        # all should have the same mask value = 1.0 - 0.5 = 0.5
        for i in range(51, 99):
            self.assertAlmostEqual(mask[i].item(), 0.5, places=5)


class TestBatchInferenceParallel(unittest.TestCase):
    def test_single_window_single_article(self):
        codec = FakeCodec()
        model = FakeModel()
        articles = [list(range(100))]  # char_ids are just 0..99
        windows = [WindowSpec(article_idx=0, start=0, end=100, responsible_start=0, responsible_end=100)]
        labels, calls = batch_inference_parallel(windows, articles, model, codec, torch.device("cpu"), batch_size=1)
        self.assertEqual(calls, 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels[0]), 100)
        # FakeModel predicts PERIOD at positions 49, 99; COMMA at 9, 19, 29, 39, 59, 69, 79, 89
        self.assertEqual(labels[0][49], PERIOD)
        self.assertEqual(labels[0][99], PERIOD)
        self.assertEqual(labels[0][9], COMMA)

    def test_multiple_windows_same_article_last_window_wins(self):
        codec = FakeCodec()
        model = FakeModel()
        articles = [list(range(200))]
        windows = [
            WindowSpec(0, 0, 100, 0, 80),
            WindowSpec(0, 80, 180, 80, 160),
            WindowSpec(0, 160, 200, 160, 200),
        ]
        labels, calls = batch_inference_parallel(windows, articles, model, codec, torch.device("cpu"), batch_size=2)
        self.assertEqual(len(labels[0]), 200)
        # positions 0..79 are labelled by window 0
        # positions 80..159 by window 1
        # positions 160..199 by window 2
        # FakeModel: PERIOD at 49, 99, 149, 199
        self.assertEqual(labels[0][49], PERIOD)
        self.assertEqual(labels[0][149], PERIOD)

    def test_multiple_articles_batched_together(self):
        codec = FakeCodec()
        model = FakeModel()
        articles = [list(range(150)), list(range(1000, 1100))]  # lengths 150, 100
        windows = [
            WindowSpec(0, 0, 150, 0, 150),
            WindowSpec(1, 0, 100, 0, 100),
        ]
        labels, calls = batch_inference_parallel(windows, articles, model, codec, torch.device("cpu"), batch_size=2)
        self.assertEqual(calls, 1)  # both windows in one batch
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(labels[0]), 150)
        self.assertEqual(len(labels[1]), 100)
        # article 0: PERIOD at 49, 99, 149
        self.assertEqual(labels[0][49], PERIOD)
        # article 1 char_ids are 1000..1099; 1049 % 50 == 49, so position 49 in article 1 should be PERIOD
        self.assertEqual(labels[1][49], PERIOD)

    def test_batch_size_splits_windows_correctly(self):
        codec = FakeCodec()
        model = FakeModel()
        articles = [list(range(100)) for _ in range(5)]
        windows = [WindowSpec(i, 0, 100, 0, 100) for i in range(5)]
        labels, calls = batch_inference_parallel(windows, articles, model, codec, torch.device("cpu"), batch_size=2)
        # 5 windows, batch_size=2 -> 3 calls (2+2+1)
        self.assertEqual(calls, 3)
        self.assertEqual(len(labels), 5)


class TestShortTailWindows(unittest.TestCase):
    def test_window_in_tail_margin_gets_extended_to_end(self):
        articles = [list(range(500))]
        windows = [
            WindowSpec(0, 0, 100, 0, 100),
            WindowSpec(0, 100, 200, 100, 200),
            WindowSpec(0, 200, 300, 200, 300),
            WindowSpec(0, 300, 400, 300, 400),
            WindowSpec(0, 400, 470, 400, 470),  # right edge at 470, within tail_margin=50 of 500
        ]
        new_windows = apply_short_tail_windows(windows, articles, tail_margin=50)
        # first 4 windows unchanged
        self.assertEqual(new_windows[0].end, 100)
        self.assertEqual(new_windows[3].end, 400)
        # last window's end extended to 500, but responsible_end stays 470
        self.assertEqual(new_windows[4].end, 500)
        self.assertEqual(new_windows[4].responsible_end, 470)

    def test_window_already_at_end_unchanged(self):
        articles = [list(range(500))]
        windows = [WindowSpec(0, 400, 500, 400, 500)]
        new_windows = apply_short_tail_windows(windows, articles, tail_margin=50)
        self.assertEqual(new_windows[0].end, 500)

    def test_window_well_before_tail_unchanged(self):
        articles = [list(range(500))]
        windows = [WindowSpec(0, 0, 100, 0, 100)]
        new_windows = apply_short_tail_windows(windows, articles, tail_margin=50)
        self.assertEqual(new_windows[0].end, 100)

    def test_multiple_articles(self):
        articles = [list(range(300)), list(range(600))]
        windows = [
            WindowSpec(0, 200, 270, 200, 270),  # article 0, within tail of 300
            WindowSpec(1, 500, 570, 500, 570),  # article 1, within tail of 600
        ]
        new_windows = apply_short_tail_windows(windows, articles, tail_margin=50)
        self.assertEqual(new_windows[0].end, 300)
        self.assertEqual(new_windows[1].end, 600)


if __name__ == "__main__":
    unittest.main()
