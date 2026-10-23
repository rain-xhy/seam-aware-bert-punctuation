"""Tests for punct.calibrate: temperature and bias fitting."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from punct.calibrate import apply_calibration, calibrate_temperature_and_bias


class TestCalibrate(unittest.TestCase):
    def test_calibrate_returns_valid_temperature_and_bias(self):
        # any logits and labels should produce a valid T (positive float) and
        # b (array of correct shape), without crashing
        logits = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 10.0],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        T, b = calibrate_temperature_and_bias(logits, labels, n_classes=3, steps=50)
        self.assertIsInstance(T, float)
        self.assertGreater(T, 0.0)
        self.assertEqual(b.shape, (3,))

    def test_apply_calibration_rescales_logits(self):
        logits = torch.tensor([[10.0, 5.0, 0.0], [0.0, 10.0, 5.0]])
        T = 2.0
        b = np.array([1.0, 0.0, -1.0])
        calibrated = apply_calibration(logits, T, b)
        # calibrated[0, 0] = (10 + 1) / 2 = 5.5
        # calibrated[0, 1] = (5 + 0) / 2 = 2.5
        # calibrated[0, 2] = (0 - 1) / 2 = -0.5
        self.assertAlmostEqual(calibrated[0, 0].item(), 5.5, places=5)
        self.assertAlmostEqual(calibrated[0, 1].item(), 2.5, places=5)
        self.assertAlmostEqual(calibrated[0, 2].item(), -0.5, places=5)

    def test_calibrate_does_not_crash_on_single_class(self):
        # all labels are the same class
        logits = torch.tensor([[10.0, 0.0], [9.0, 1.0], [8.0, 2.0]])
        labels = torch.tensor([0, 0, 0])
        T, b = calibrate_temperature_and_bias(logits, labels, n_classes=2, steps=10)
        self.assertIsInstance(T, float)
        self.assertEqual(b.shape, (2,))


if __name__ == "__main__":
    unittest.main()
