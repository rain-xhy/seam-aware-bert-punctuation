import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_seeds import aggregate, extract


class AggregateSeedsTests(unittest.TestCase):
    def test_extract_methods_and_configs(self):
        a = {"methods": {"x": {"quality": {"micro_f1": 0.8}}}}
        b = {"configs": {"y": {"all": {"micro_f1": 0.7}}}}
        self.assertEqual(extract(a, "micro_f1"), {"x": 0.8})
        self.assertEqual(extract(b, "micro_f1"), {"y": 0.7})
        with self.assertRaises(ValueError):
            extract({}, "micro_f1")

    def test_mean_std_sample(self):
        r = aggregate([{"m": 0.8}, {"m": 0.9}, {"m": 1.0}])["m"]
        self.assertAlmostEqual(r["mean"], 0.9)
        self.assertAlmostEqual(r["std"], 0.1)  # sample std, n-1
        self.assertEqual(r["n"], 3)

    def test_single_run_has_no_std(self):
        r = aggregate([{"m": 0.8}])["m"]
        self.assertIsNone(r["std"])

    def test_method_missing_in_some_files(self):
        r = aggregate([{"a": 0.5, "b": 0.6}, {"a": 0.7}])
        self.assertEqual(r["a"]["n"], 2)
        self.assertEqual(r["b"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
