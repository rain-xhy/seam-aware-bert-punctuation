import unittest

import numpy as np
import torch

from punct.edep_scan import edep_scan

CLS, SEP, PAD = 101, 102, 0
DEV = torch.device("cpu")
LEFT_BAD, RIGHT_BAD = 6, 2


def edge_sensitive_logits(input_ids, attention_mask):
    """Correct (id % 8) except the first LEFT_BAD chars and last RIGHT_BAD chars
    of every window, which are shifted to a wrong class."""
    labels = (input_ids % 8).clone()
    for b in range(input_ids.shape[0]):
        n = int(attention_mask[b].sum().item()) - 2  # chars in this row
        for p in range(1, 1 + n):
            if p <= LEFT_BAD or p > n - RIGHT_BAD:
                labels[b, p] = (labels[b, p] + 1) % 8
    return torch.nn.functional.one_hot(labels, 8).float() * 10.0


class EdepScanTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.articles = [rng.integers(200, 5000, size=n).tolist() for n in (600, 900, 1200)]
        self.gold = [np.asarray(a) % 8 for a in self.articles]

    def scan(self, **kw):
        return edep_scan(self.articles, self.gold, edge_sensitive_logits, CLS, SEP, PAD, DEV,
                         window=100, overlap=40, max_distance=20, batch_size=4, **kw)

    def test_left_and_right_curves_show_asymmetric_edges(self):
        r = self.scan()
        left, right = np.array(r["left"]), np.array(r["right"])
        self.assertTrue((left[:LEFT_BAD] == 1.0).all())
        self.assertTrue((left[LEFT_BAD:] == 0.0).all())
        self.assertTrue((right[:RIGHT_BAD] == 1.0).all())
        self.assertTrue((right[RIGHT_BAD:] == 0.0).all())

    def test_masking_widths_match_bad_edge_sizes(self):
        r = self.scan()
        self.assertEqual(r["m_l"], LEFT_BAD)
        self.assertEqual(r["m_r"], RIGHT_BAD)
        self.assertEqual(r["centre_error"], 0.0)

    def test_article_boundaries_not_counted_as_seams(self):
        # one article that fits a single window has no internal edge at all
        r = edep_scan([self.articles[0][:80]], [self.gold[0][:80]], edge_sensitive_logits,
                      CLS, SEP, PAD, DEV, window=100, overlap=40, max_distance=10)
        self.assertEqual(sum(r["left_n"]), 0)
        self.assertEqual(sum(r["right_n"]), 0)

    def test_batch_size_does_not_change_curves(self):
        a = self.scan()
        b = edep_scan(self.articles, self.gold, edge_sensitive_logits, CLS, SEP, PAD, DEV,
                      window=100, overlap=40, max_distance=20, batch_size=1)
        self.assertEqual(a["left"], b["left"])
        self.assertEqual(a["right"], b["right"])


if __name__ == "__main__":
    unittest.main()
