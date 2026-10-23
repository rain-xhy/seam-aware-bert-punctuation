import unittest

import numpy as np

from punct.bench import agreement, time_interleaved


class BenchTests(unittest.TestCase):
    def test_interleaved_runs_each_config(self):
        calls = {"a": 0, "b": 0}

        def mk(k):
            def f():
                calls[k] += 1
                return k
            return f

        out = time_interleaved({"a": mk("a"), "b": mk("b")}, repeats=3, warmup=1)
        self.assertEqual(calls, {"a": 4, "b": 4})
        self.assertEqual(len(out["a"]["runs"]), 3)
        self.assertEqual(out["b"]["last_result"], "b")
        self.assertLessEqual(out["a"]["min"], out["a"]["median"])

    def test_agreement(self):
        a = [np.array([1, 2, 3]), np.array([4])]
        b = [np.array([1, 0, 3]), np.array([4])]
        r = agreement(a, b)
        self.assertEqual(r["n_diff"], 1)
        self.assertAlmostEqual(r["agreement"], 0.75)
        with self.assertRaises(ValueError):
            agreement(a, [np.array([1])])

    def test_zero_repeats_rejected(self):
        with self.assertRaises(ValueError):
            time_interleaved({"a": lambda: 0}, repeats=0)


if __name__ == "__main__":
    unittest.main()
