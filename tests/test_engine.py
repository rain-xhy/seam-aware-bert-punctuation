import unittest

import numpy as np
import torch

from punct.engine import decode_parallel, edge_weights, pool_windows

CLS, SEP, PAD = 101, 102, 0
DEV = torch.device("cpu")


def fake_logits_fn(input_ids, attention_mask):
    """Output depends only on the token id at each position (position-determined)."""
    labels = input_ids % 8
    return torch.nn.functional.one_hot(labels, 8).float() * 10.0


def expected(article):
    return np.array(article) % 8


def run(articles, **kw):
    return decode_parallel(articles, fake_logits_fn, CLS, SEP, PAD, DEV, **kw)


class EngineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.articles = [rng.integers(200, 5000, size=n).tolist() for n in (1, 7, 50, 51, 99, 100, 101, 333)]

    def test_matches_single_forward_hard_mask(self):
        for tail in ("short", "aligned"):
            labels, stats = run(self.articles, window=50, overlap=20, tail=tail, mask_left=5, mask_right=5, batch_size=4)
            for a, l in zip(self.articles, labels):
                np.testing.assert_array_equal(l, expected(a))
            self.assertEqual(stats.text_chars, sum(len(a) for a in self.articles))

    def test_gauss_profile_and_mixed_batches(self):
        labels, _ = run(self.articles, window=40, overlap=10, tail="short", profile="gauss", batch_size=3)
        for a, l in zip(self.articles, labels):
            np.testing.assert_array_equal(l, expected(a))

    def test_batch_size_does_not_change_result(self):
        r1, _ = run(self.articles, window=60, overlap=20, batch_size=1)
        r2, _ = run(self.articles, window=60, overlap=20, batch_size=64)
        for x, y in zip(r1, r2):
            np.testing.assert_array_equal(x, y)

    def test_empty_and_single_char(self):
        labels, stats = run([[300], [400, 401]], window=10, overlap=2)
        self.assertEqual([len(l) for l in labels], [1, 2])
        self.assertEqual(stats.n_windows, 2)

    def test_ter_short_vs_aligned(self):
        _, s_short = run([list(range(200, 1000))], window=100, overlap=30, tail="short")
        _, s_aligned = run([list(range(200, 1000))], window=100, overlap=30, tail="aligned")
        self.assertLessEqual(s_short.ter, s_aligned.ter)

    def test_excessive_mask_raises(self):
        with self.assertRaises(ValueError):
            run([list(range(200, 600))], window=100, overlap=10, mask_left=60, mask_right=60)

    def test_edge_weights_article_boundary_exempt(self):
        w = edge_weights(10, True, True, 3, 3)
        self.assertTrue((w == 1).all())
        w = edge_weights(10, False, False, 3, 2)
        self.assertEqual(list(w), [0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

    def test_oom_halves_batch(self):
        calls = {"n": 0}

        def flaky(ids, mask):
            calls["n"] += 1
            if ids.shape[0] > 2:
                raise torch.OutOfMemoryError("fake")
            return fake_logits_fn(ids, mask)

        art = [list(range(200, 800))]
        labels, stats = decode_parallel(art, flaky, CLS, SEP, PAD, DEV, window=50, overlap=10, batch_size=8)
        np.testing.assert_array_equal(labels[0], expected(art[0]))
        self.assertGreaterEqual(stats.n_oom_retries, 2)
        self.assertEqual(stats.batch_size_used, 2)

    def test_pool_windows_cover(self):
        pw = pool_windows([0, 5, 250], 100, 20, "short")
        self.assertEqual(sum(1 for w in pw if w.article == 0), 0)


if __name__ == "__main__":
    unittest.main()
