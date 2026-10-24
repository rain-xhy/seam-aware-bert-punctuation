import unittest

import numpy as np
import torch

from punct.pipeline import decode_two_pass_pipelined
from punct.two_pass import decode_two_pass

SPECIAL = (101, 102, 0)
DEV = torch.device("cpu")


def fake_fn(input_ids, attention_mask):
    return torch.nn.functional.one_hot(input_ids % 8, 8).float() * 10.0


def low_conf_fn(input_ids, attention_mask):
    a = torch.nn.functional.one_hot(input_ids % 8, 8).float()
    b = torch.nn.functional.one_hot((input_ids + 1) % 8, 8).float()
    return (a + b) * 10.0


def mixed_fn(input_ids, attention_mask):
    """Confident on even ids, a tie between two classes on odd ids."""
    a = torch.nn.functional.one_hot(input_ids % 8, 8).float()
    b = torch.nn.functional.one_hot((input_ids + 1) % 8, 8).float()
    odd = (input_ids % 2 == 1).float().unsqueeze(-1)
    return (a + b * odd) * 10.0


class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.arts = [rng.integers(200, 5000, size=n).tolist() for n in (0, 1, 30, 400, 401, 950, 1500, 2600)]

    def check_equal(self, fn, **kw):
        ref, info_ref = decode_two_pass(self.arts, fn, SPECIAL, DEV, **kw)
        got, info = decode_two_pass_pipelined(self.arts, fn, SPECIAL, DEV, **kw)
        for a, (x, y) in enumerate(zip(ref, got)):
            np.testing.assert_array_equal(x, y, err_msg=f"article {a}")
        for k in ("n_selected", "n_patch_windows", "patch_tokens", "n_changed"):
            self.assertEqual(info_ref[k], info[k], k)
        self.assertAlmostEqual(info_ref["ter"], info["ter"])
        return info

    def test_matches_two_pass_when_nothing_selected(self):
        info = self.check_equal(fake_fn, conf_thr=0.9)
        self.assertEqual(info["n_selected"], 0)

    def test_matches_two_pass_with_patches(self):
        info = self.check_equal(low_conf_fn, conf_thr=0.9)
        self.assertGreater(info["n_patch_windows"], 0)

    def test_matches_with_mixed_confidence_and_small_batches(self):
        info = self.check_equal(mixed_fn, conf_thr=0.9, batch_size=3)
        self.assertGreater(info["n_selected"], 0)

    def test_matches_with_calibration_and_gauss_profile(self):
        cal = (1.5, np.zeros(8))
        self.check_equal(mixed_fn, conf_thr=0.9, calibration=cal, profile="gauss")

    def test_oom_halves_batch(self):
        def flaky(ids, mask):
            if ids.shape[0] > 2:
                raise torch.OutOfMemoryError("fake")
            return low_conf_fn(ids, mask)

        ref, _ = decode_two_pass(self.arts, low_conf_fn, SPECIAL, DEV)
        got, info = decode_two_pass_pipelined(self.arts, flaky, SPECIAL, DEV, batch_size=8)
        self.assertGreaterEqual(info["n_oom_retries"], 2)
        for x, y in zip(ref, got):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
