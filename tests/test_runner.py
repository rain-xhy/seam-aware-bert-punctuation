import unittest

import numpy as np
import torch

from punct.runner import exact_agreement, run_method, score

CLS, SEP, PAD = 101, 102, 0
DEV = torch.device("cpu")


def fake_logits_fn(input_ids, attention_mask):
    return torch.nn.functional.one_hot(input_ids % 8, 8).float() * 10.0


class RunnerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.arts = [rng.integers(200, 5000, size=n).astype(np.int64) for n in (30, 130, 450, 900)]
        self.gold = [a % 8 for a in self.arts]

    def test_all_methods_recover_position_determined_labels(self):
        sp = (CLS, SEP, PAD)
        for name, kw in [
            ("parallel", dict(window=100, overlap=20, tail="short", batch_size=4)),
            ("direct_cut", dict(window=64)),
            ("serial_teacher", {}), ("serial_yu", {}), ("serial_tang", {}), ("serial_cross", {}),
        ]:
            labels, stats = run_method(name, self.arts, fake_logits_fn, sp, DEV, **kw)
            self.assertIn("seconds", stats)
            for g, p in zip(self.gold, labels):
                np.testing.assert_array_equal(g, p, err_msg=name)

    def test_score_perfect_and_agreement(self):
        s = score(self.gold, self.gold, seam_stride=100)
        self.assertAlmostEqual(s["micro_f1"], 1.0)
        self.assertEqual(s["errors"]["total_errors"], 0)
        ag = exact_agreement(self.gold, self.gold)
        self.assertEqual(ag["n_diff"], 0)

    def test_score_counts_errors(self):
        pred = [g.copy() for g in self.gold]
        pred[0][0] = (pred[0][0] + 1) % 8
        s = score(self.gold, pred)
        self.assertEqual(exact_agreement(self.gold, pred)["n_diff"], 1)
        self.assertLess(s["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
