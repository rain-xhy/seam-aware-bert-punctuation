"""Tests for punct.merge, using a fake model instead of a real BERT.

The key property the plan calls for: merging predictions from overlapping
windows of a fake model that gives a DETERMINISTIC, position-dependent
probability should reproduce exactly what a single pass over the whole
article (one window covering everything) would give — because the fake
model's output does not depend on window boundaries at all. This isolates
bugs in the merge arithmetic from anything about BERT itself.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from punct.merge import (
    WindowProbs,
    gaussian_weight,
    merge_hard_mask,
    merge_position_weighted,
    merge_tang_drop_last_sentence,
)
from punct.window_plan import Window, plan_windows

NUM_LABELS = 8


def fake_model_probs(global_positions: np.ndarray) -> np.ndarray:
    """A deterministic fake model: the predicted label only depends on the
    GLOBAL character position (position % 8), with probability mass 0.9 on
    that label and the rest split evenly. Crucially, it does NOT depend on
    where the window starts or how long it is, which lets a correct merge
    reconstruct exactly what one full-length pass would produce."""
    n = len(global_positions)
    probs = np.full((n, NUM_LABELS), 0.1 / (NUM_LABELS - 1), dtype=np.float64)
    winners = global_positions % NUM_LABELS
    probs[np.arange(n), winners] = 0.9
    return probs


def run_fake_model_over_windows(windows, article_length):
    out = []
    for w in windows:
        positions = np.arange(w.start, w.end)
        out.append(WindowProbs(window=w, probs=fake_model_probs(positions)))
    return out


class TestMergeHardMaskReconstructsSinglePass(unittest.TestCase):
    def test_matches_single_window_pass_various_lengths(self):
        for length in (50, 300, 511, 1000, 9592):
            for window, overlap in ((510, 20), (125, 20), (64, 0)):
                if overlap >= window:
                    continue
                windows = plan_windows(length, window, overlap, tail="short")
                window_probs = run_fake_model_over_windows(windows, length)
                merged = merge_hard_mask(window_probs, length, NUM_LABELS, mask_left=0, mask_right=0)
                reference = fake_model_probs(np.arange(length))
                np.testing.assert_allclose(merged, reference, atol=1e-9,
                                            err_msg=f"length={length} window={window} overlap={overlap}")

    def test_single_window_whole_article_is_a_noop(self):
        length = 300
        windows = [Window(0, length)]
        window_probs = run_fake_model_over_windows(windows, length)
        merged = merge_hard_mask(window_probs, length, NUM_LABELS, mask_left=0, mask_right=0)
        reference = fake_model_probs(np.arange(length))
        np.testing.assert_allclose(merged, reference, atol=1e-9)


class TestMergeHardMaskWithRealMasking(unittest.TestCase):
    def test_masked_overlap_still_reconstructs_fake_model_exactly(self):
        # because the fake model is position-only, masking out low-context
        # edges should STILL reconstruct the single-pass reference exactly:
        # the unmasked region of each window already agrees with it.
        length = 2000
        window, overlap = 510, 64
        windows = plan_windows(length, window, overlap, tail="short")
        window_probs = run_fake_model_over_windows(windows, length)
        merged = merge_hard_mask(window_probs, length, NUM_LABELS, mask_left=20, mask_right=20)
        reference = fake_model_probs(np.arange(length))
        np.testing.assert_allclose(merged, reference, atol=1e-9)

    def test_mask_too_wide_for_overlap_raises(self):
        length = 2000
        window, overlap = 510, 20
        windows = plan_windows(length, window, overlap, tail="short")
        window_probs = run_fake_model_over_windows(windows, length)
        # mask_left + mask_right (30+30=60) > overlap (20): some position
        # is masked out of every window that covers it
        with self.assertRaises(ValueError):
            merge_hard_mask(window_probs, length, NUM_LABELS, mask_left=30, mask_right=30)

    def test_article_true_edges_are_never_masked(self):
        # a short article entirely within one window: masking must not
        # remove the article's own first/last character even though they
        # are at the window's edge
        length = 10
        windows = [Window(0, length)]
        window_probs = run_fake_model_over_windows(windows, length)
        merged = merge_hard_mask(window_probs, length, NUM_LABELS, mask_left=5, mask_right=5)
        reference = fake_model_probs(np.arange(length))
        np.testing.assert_allclose(merged, reference, atol=1e-9)


class TestMergePositionWeighted(unittest.TestCase):
    def test_gaussian_weight_reconstructs_fake_model_exactly(self):
        length = 2000
        window, overlap = 510, 64
        windows = plan_windows(length, window, overlap, tail="short")
        window_probs = run_fake_model_over_windows(windows, length)
        merged = merge_position_weighted(window_probs, length, NUM_LABELS, gaussian_weight())
        reference = fake_model_probs(np.arange(length))
        np.testing.assert_allclose(merged, reference, atol=1e-9)

    def test_weight_fn_must_be_positive(self):
        length = 300
        windows = [Window(0, length)]
        window_probs = run_fake_model_over_windows(windows, length)
        with self.assertRaises(ValueError):
            merge_position_weighted(window_probs, length, NUM_LABELS, lambda i, n: 0.0)


class TestMergeTangDropLastSentence(unittest.TestCase):
    def test_every_position_filled_exactly_once(self):
        # sentence-final label ids: say labels 1 and 2 are sentence-final
        length = 400
        window, overlap = 125, 20
        windows = plan_windows(length, window, overlap, tail="short")
        predicted = []
        for w in windows:
            positions = np.arange(w.start, w.end)
            predicted.append((positions % NUM_LABELS).astype(np.int64))
        merged = merge_tang_drop_last_sentence(
            [WindowProbs(w, fake_model_probs(np.arange(w.start, w.end))) for w in windows],
            length, NUM_LABELS, predicted, sentence_final_label_ids=(1, 2),
        )
        self.assertEqual(merged.sum(), length)  # exactly one one-hot per position

    def test_single_window_keeps_everything(self):
        length = 50
        windows = [Window(0, length)]
        predicted = [(np.arange(length) % NUM_LABELS).astype(np.int64)]
        merged = merge_tang_drop_last_sentence(
            [WindowProbs(windows[0], fake_model_probs(np.arange(length)))],
            length, NUM_LABELS, predicted, sentence_final_label_ids=(1, 2),
        )
        self.assertEqual(merged.sum(), length)


if __name__ == "__main__":
    unittest.main()
