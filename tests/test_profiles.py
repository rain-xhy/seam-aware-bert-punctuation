import unittest

import numpy as np

from punct.profiles import edep_profile, gauss_profile, hard_profile, linear_decay_profile


class ProfileTests(unittest.TestCase):
    def test_hard_profile_masks_edges(self):
        fn = hard_profile(mask_left=3, mask_right=2)
        w = fn(10, at_article_start=False, at_article_end=False)
        # Positions 0,1,2 masked (left), 8,9 masked (right)
        self.assertEqual(w[:3].sum(), 0.0)
        self.assertEqual(w[8:].sum(), 0.0)
        self.assertEqual(w[3:8].min(), 1.0)

    def test_hard_profile_exempts_article_boundaries(self):
        fn = hard_profile(mask_left=3, mask_right=2)
        w_start = fn(10, at_article_start=True, at_article_end=False)
        self.assertEqual(w_start[:3].sum(), 3.0)  # not masked
        w_end = fn(10, at_article_start=False, at_article_end=True)
        self.assertEqual(w_end[8:].sum(), 2.0)  # not masked

    def test_gauss_profile_peaks_at_center(self):
        fn = gauss_profile(sigma_ratio=0.25)
        w = fn(100, False, False)
        self.assertEqual(w.argmax(), 49)  # center = (100-1)/2
        self.assertGreater(w[49], 0.9)
        self.assertLess(w[0], 0.2)

    def test_edep_profile_uses_error_curves(self):
        left_err = [0.5, 0.3, 0.1, 0.05]
        right_err = [0.4, 0.2, 0.1, 0.05]
        fn = edep_profile(left_err, right_err, floor=0.1, ceiling=1.0)
        w = fn(10, False, False)

        # Left edge: weight = 1.0 - error
        self.assertAlmostEqual(w[0], 0.5, places=2)  # 1.0 - 0.5
        self.assertAlmostEqual(w[1], 0.7, places=2)  # 1.0 - 0.3

        # Right edge
        self.assertAlmostEqual(w[9], 0.6, places=2)  # 1.0 - 0.4

        # Middle: should be near ceiling
        self.assertGreater(w[5], 0.9)

    def test_edep_profile_respects_floor(self):
        left_err = [0.95, 0.5]  # Very high error
        fn = edep_profile(left_err, [0.1], floor=0.2, ceiling=1.0)
        w = fn(5, False, False)
        # 1.0 - 0.95 = 0.05, clamped to floor 0.2
        self.assertAlmostEqual(w[0], 0.2, places=2)

    def test_linear_decay_profile(self):
        fn = linear_decay_profile(threshold_left=4, threshold_right=3)
        w = fn(10, False, False)

        # Left: linear from 0 to 1 over 4 positions
        self.assertAlmostEqual(w[0], 0.0, places=2)
        self.assertAlmostEqual(w[1], 0.25, places=2)
        self.assertAlmostEqual(w[2], 0.5, places=2)
        self.assertAlmostEqual(w[3], 0.75, places=2)

        # Middle: full weight
        self.assertEqual(w[5], 1.0)

        # Right: linear from 1 to 0 over 3 positions
        self.assertAlmostEqual(w[9], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
