import unittest

import numpy as np
import torch

from punct.two_pass import decode_two_pass, plan_patches, seam_candidates, select_positions, _anchor_start
from punct.engine import pool_windows

SPECIAL = (101, 102, 0)
DEV = torch.device("cpu")


def fake_fn(input_ids, attention_mask):
    return torch.nn.functional.one_hot(input_ids % 8, 8).float() * 10.0


def low_conf_fn(input_ids, attention_mask):
    # ties between two classes -> top-1 prob 0.5 everywhere
    a = torch.nn.functional.one_hot(input_ids % 8, 8).float()
    b = torch.nn.functional.one_hot((input_ids + 1) % 8, 8).float()
    return (a + b) * 10.0


class TwoPassTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.arts = [rng.integers(200, 5000, size=n).tolist() for n in (30, 400, 401, 1500, 2600)]

    def test_candidates_exclude_article_ends(self):
        pooled = pool_windows([1000], 400, 100, "short")
        m = seam_candidates(pooled, [1000], 8)[0]
        self.assertFalse(m[:8].any())
        self.assertFalse(m[-8:].any())
        self.assertTrue(m[300:308].any() or m[392:400].any())

    def test_confidence_mode_selects_nothing_when_confident(self):
        labels, info = decode_two_pass(self.arts, fake_fn, SPECIAL, DEV, conf_thr=0.9)
        self.assertEqual(info["n_selected"], 0)
        for a, l in zip(self.arts, labels):
            np.testing.assert_array_equal(l, np.array(a) % 8)

    def test_low_confidence_triggers_patches_and_covers_positions(self):
        labels, info = decode_two_pass(self.arts, low_conf_fn, SPECIAL, DEV, conf_thr=0.9)
        self.assertGreater(info["n_selected"], 0)
        self.assertGreater(info["n_patch_windows"], 0)
        self.assertGreater(info["ter"], info["pass1_ter"])
        for a, l in zip(self.arts, labels):
            self.assertEqual(len(l), len(a))

    def test_plan_patches_bounds_and_margins(self):
        n = 3000
        pos = {0: np.array([5, 6, 399, 400, 1200, 2995])}
        labels = [np.zeros(n, dtype=np.int64)]
        patches = plan_patches(pos, [n], labels, patch_window=400, edge_margin=128, anchor_range=0)
        seen = []
        for p in patches:
            self.assertTrue(0 <= p.start < p.end <= n)
            for q in p.positions:
                self.assertTrue(p.start <= q < p.end)
                if p.start > 0:
                    self.assertGreaterEqual(q - p.start, 128)
                if p.end < n:
                    self.assertGreaterEqual(p.end - 1 - q, 127)
            seen += p.positions
        self.assertEqual(sorted(seen), [5, 6, 399, 400, 1200, 2995])

    def test_anchoring_near_article_end_still_covers_position(self):
        # regression: a sentence mark just left of the last window start made the
        # anchored window end before the selected position (random control crash)
        n = 3000
        lab = np.zeros(n, dtype=np.int64)
        lab[n - 405] = 2
        pos = {0: np.array([n - 10])}
        patches = plan_patches(pos, [n], [lab], patch_window=400, edge_margin=128, anchor_range=16)
        self.assertEqual(sum(len(p.positions) for p in patches), 1)
        p = patches[0]
        self.assertTrue(p.start <= n - 10 < p.end)

    def test_short_article_uses_whole_article(self):
        patches = plan_patches({0: np.array([10, 20])}, [50], [np.zeros(50, dtype=np.int64)])
        self.assertEqual((patches[0].start, patches[0].end), (0, 50))
        self.assertEqual(patches[0].positions, [10, 20])

    def test_anchor_moves_after_sentence_mark(self):
        lab = np.zeros(100, dtype=np.int64)
        lab[40] = 2
        self.assertEqual(_anchor_start(50, lab, 16), 41)
        self.assertEqual(_anchor_start(50, lab, 5), 50)
        self.assertEqual(_anchor_start(0, lab, 16), 0)

    def test_random_control_matches_budget(self):
        probs = [np.full((n, 8), 0.125) for n in (500, 700)]
        masks = {0: np.ones(500, dtype=bool), 1: np.ones(700, dtype=bool)}
        sel = select_positions(probs, masks, 0.9, "random", np.random.default_rng(0), 50)
        self.assertEqual(sum(len(v) for v in sel.values()), 50)

    def test_empty_selection_and_no_patch(self):
        labels, info = decode_two_pass(self.arts, fake_fn, SPECIAL, DEV, patch=False)
        self.assertEqual(info["ter"], info["pass1_ter"])
        self.assertEqual(len(labels), len(self.arts))


if __name__ == "__main__":
    unittest.main()
