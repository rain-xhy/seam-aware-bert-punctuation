import unittest

import numpy as np
import torch

from punct.engine import PooledWindow
from punct.tang_parallel import decode_tang_parallel, merge_drop_last_sentence

DEV = torch.device("cpu")
SPECIAL = (101, 102, 0)


def fake_fn(input_ids, attention_mask):
    return torch.nn.functional.one_hot(input_ids % 8, 8).float() * 10.0


class TangParallelTests(unittest.TestCase):
    def test_position_determined_model_matches_single_forward(self):
        rng = np.random.default_rng(0)
        arts = [rng.integers(200, 5000, size=n).tolist() for n in (1, 30, 125, 126, 400, 1000)]
        labels, stats, info = decode_tang_parallel(arts, fake_fn, SPECIAL, DEV, window=125, overlap=20, batch_size=5)
        for a, l in zip(arts, labels):
            np.testing.assert_array_equal(l, np.array(a) % 8)
        self.assertEqual(stats.text_chars, sum(len(a) for a in arts))

    def test_drops_tail_after_last_sentence_mark(self):
        pooled = [PooledWindow(0, 0, 10), PooledWindow(0, 6, 16)]
        a = np.array([1, 1, 2, 1, 1, 1, 1, 1, 1, 1])      # last '。' at local 2 -> keep 3
        b = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        out, gap = merge_drop_last_sentence([16], pooled, [a, b])
        np.testing.assert_array_equal(out[0][:3], [1, 1, 2])
        # positions 3..5 are kept by no window (window 1 dropped them, window 2 starts at 6):
        # they are filled from the first covering window and counted as gap
        np.testing.assert_array_equal(out[0][3:6], [1, 1, 1])
        np.testing.assert_array_equal(out[0][6:], [5] * 10)
        self.assertEqual(gap, 3)

    def test_no_sentence_mark_keeps_whole_window(self):
        pooled = [PooledWindow(0, 0, 10), PooledWindow(0, 8, 16)]
        a = np.ones(10, dtype=np.int64)
        b = np.full(8, 3, dtype=np.int64)
        out, gap = merge_drop_last_sentence([16], pooled, [a, b])
        self.assertEqual(gap, 0)
        np.testing.assert_array_equal(out[0], [1] * 10 + [3] * 6)


if __name__ == "__main__":
    unittest.main()
