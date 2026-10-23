import unittest

import numpy as np

from punct.metrics import punctuation_prf
from scripts.recompute import compare, prf_from_scratch


class RecomputeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.gold = rng.choice(8, size=5000, p=[0.8, 0.07, 0.05, 0.01, 0.01, 0.02, 0.02, 0.02])
        flip = rng.random(5000) < 0.15
        self.pred = np.where(flip, rng.integers(0, 8, size=5000), self.gold)

    def test_matches_package_metrics(self):
        mine = prf_from_scratch(self.gold, self.pred)
        ref = punctuation_prf(self.gold, self.pred, 8, 0)
        for k in ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "accuracy"):
            self.assertAlmostEqual(mine[k], ref[k], places=12)

    def test_compare_ok_and_detects_tampering(self):
        stored = prf_from_scratch(self.gold, self.pred)
        result = {"methods": {"m": {"quality": dict(stored)}}}
        out = compare(result, {"m": self.pred}, self.gold)
        self.assertEqual(out["mismatches"], [])
        self.assertEqual(out["checked"], 5)
        result["methods"]["m"]["quality"]["micro_f1"] += 0.01
        out = compare(result, {"m": self.pred}, self.gold)
        self.assertEqual(len(out["mismatches"]), 1)

    def test_missing_and_wrong_length(self):
        result = {"methods": {"a": {"quality": {}}, "b": {"quality": {}}}}
        out = compare(result, {"b": self.pred[:10]}, self.gold)
        self.assertEqual(len(out["mismatches"]), 2)


if __name__ == "__main__":
    unittest.main()
