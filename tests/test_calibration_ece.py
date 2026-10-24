import unittest

import numpy as np

from punct.calibrate import calibrate_probs, expected_calibration_error
from punct.two_pass import decode_two_pass
import torch

SPECIAL = (101, 102, 0)
DEV = torch.device("cpu")


def confident_fn(ids, mask):
    return torch.nn.functional.one_hot(ids % 8, 8).float() * 10.0


class CalibrationTests(unittest.TestCase):
    def test_ece_perfect_and_overconfident(self):
        probs = np.eye(4)[[0, 1, 2, 3]]
        self.assertAlmostEqual(expected_calibration_error(probs, np.array([0, 1, 2, 3]))["ece"], 0.0)
        # always 100% confident but only half right -> ECE 0.5
        r = expected_calibration_error(probs, np.array([0, 1, 0, 0]))
        self.assertAlmostEqual(r["ece"], 0.5)
        self.assertEqual(sum(b["count"] for b in r["bins"]), 4)

    def test_calibrate_probs_identity_and_argmax(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(8), size=50)
        np.testing.assert_allclose(calibrate_probs(p, 1.0, np.zeros(8)), p, atol=1e-6)
        q = calibrate_probs(p, 2.5, np.zeros(8))
        np.testing.assert_array_equal(q.argmax(1), p.argmax(1))   # temperature keeps argmax
        np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-6)
        self.assertTrue((q.max(1) <= p.max(1) + 1e-9).all())     # T>1 softens

    def test_two_pass_calibration_changes_selection_not_pass1(self):
        rng = np.random.default_rng(2)
        arts = [rng.integers(200, 5000, size=n).tolist() for n in (900, 1500)]
        _, base = decode_two_pass(arts, confident_fn, SPECIAL, DEV, conf_thr=0.9)
        # strong softening pushes every confidence below the threshold
        lab, info = decode_two_pass(arts, confident_fn, SPECIAL, DEV, conf_thr=0.9,
                                    calibration=(50.0, np.zeros(8)))
        self.assertEqual(base["n_selected"], 0)
        self.assertGreater(info["n_selected"], 0)
        self.assertEqual(info["pass1_tokens"], base["pass1_tokens"])


if __name__ == "__main__":
    unittest.main()
