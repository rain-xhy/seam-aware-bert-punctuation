"""Edge cases from 完善方案.md section 8 that no other test file covers:
all-[UNK] text, OOM that cannot be halved further, and a 100000-character
article through the real decoding engine (position-determined fake model)."""
import unittest

import numpy as np
import torch

from punct.engine import decode_parallel

CLS, SEP, PAD, UNK = 101, 102, 0, 100
DEV = torch.device("cpu")


def fake_fn(input_ids, attention_mask):
    return torch.nn.functional.one_hot(input_ids % 8, 8).float() * 10.0


class EdgeCaseTests(unittest.TestCase):
    def test_all_unk_article(self):
        art = [UNK] * 1234
        labels, stats = decode_parallel([art], fake_fn, CLS, SEP, PAD, DEV, window=400, overlap=100)
        self.assertEqual(len(labels[0]), 1234)
        np.testing.assert_array_equal(labels[0], np.full(1234, UNK % 8))
        self.assertEqual(stats.text_chars, 1234)

    def test_oom_at_batch_one_raises(self):
        def always_oom(ids, mask):
            raise torch.OutOfMemoryError("fake")

        with self.assertRaises(torch.OutOfMemoryError):
            decode_parallel([list(range(200, 900))], always_oom, CLS, SEP, PAD, DEV,
                            window=100, overlap=20, batch_size=4)

    def test_hundred_thousand_chars(self):
        rng = np.random.default_rng(3)
        art = rng.integers(200, 5000, size=100000).tolist()
        for tail in ("short", "aligned"):
            labels, stats = decode_parallel([art], fake_fn, CLS, SEP, PAD, DEV, window=400,
                                            overlap=100, tail=tail, batch_size=16)
            np.testing.assert_array_equal(labels[0], np.array(art) % 8)
            self.assertEqual(stats.text_chars, 100000)


if __name__ == "__main__":
    unittest.main()
