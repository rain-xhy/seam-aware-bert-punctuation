import unittest
import numpy as np
from punct.metrics import (error_decomposition, seam_f1_fixed_positions,
                           fit_char_frequency_baseline, predict_char_frequency_baseline)


class TestExtra(unittest.TestCase):
    def test_error_decomposition(self):
        gold = np.array([0, 1, 2, 0, 3, 0])
        pred = np.array([0, 0, 1, 4, 3, 0])
        r = error_decomposition(gold, pred)
        self.assertEqual((r["missed"], r["spurious"], r["confused"]), (1, 1, 1))
        self.assertEqual(r["total_errors"], 3)
        self.assertEqual(r["gold_marks"], 3)

    def test_error_decomposition_shape(self):
        with self.assertRaises(ValueError):
            error_decomposition(np.zeros(3, int), np.zeros(4, int))

    def test_seam_fixed_positions(self):
        n = 40
        gold = np.zeros(n, int); pred = np.zeros(n, int)
        gold[19] = 1; pred[19] = 1   # inside seam around 20 (margin 3: 17..22)
        gold[5] = 1                  # interior miss
        r = seam_f1_fixed_positions(gold, pred, [20], margin=3)
        self.assertEqual(r["seam_positions"], 6)
        self.assertEqual(r["seam_f1"], 1.0)
        self.assertEqual(r["interior_f1"], 0.0)
        self.assertEqual(r["seam_support"], 1)
        self.assertEqual(r["interior_support"], 1)

    def test_seam_boundary_clipping(self):
        gold = np.zeros(5, int); pred = np.zeros(5, int)
        r = seam_f1_fixed_positions(gold, pred, [0, 100], margin=2)
        self.assertEqual(r["seam_positions"], 2)

    def test_frequency_baseline(self):
        chars = [np.array([7, 8, 7]), np.array([7, 8])]
        labels = [np.array([1, 0, 1]), np.array([2, 0])]
        t = fit_char_frequency_baseline(chars, labels)
        self.assertEqual(t[7], 1)
        self.assertEqual(t[8], 0)
        p = predict_char_frequency_baseline(t, np.array([7, 8, 99]))
        self.assertEqual(p.tolist(), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
