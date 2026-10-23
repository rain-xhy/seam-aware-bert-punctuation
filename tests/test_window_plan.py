"""Tests for punct.window_plan — pure functions, no model/GPU needed."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.window_plan import Window, coverage_ok, plan_windows, total_forward_chars


class TestPlanWindowsBasics(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(plan_windows(0, 510, 20), [])

    def test_single_char(self):
        windows = plan_windows(1, 510, 20)
        self.assertEqual(windows, [Window(0, 1)])

    def test_exactly_window_length(self):
        windows = plan_windows(510, 510, 20)
        self.assertEqual(windows, [Window(0, 510)])

    def test_one_more_than_window_length_aligned(self):
        windows = plan_windows(511, 510, 20, tail="aligned")
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0], Window(0, 510))
        self.assertEqual(windows[-1].end, 511)
        self.assertEqual(windows[-1].end - windows[-1].start, 510)  # aligned tail is full length

    def test_one_more_than_window_length_short(self):
        windows = plan_windows(511, 510, 20, tail="short")
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[-1].end, 511)
        self.assertLess(windows[-1].end - windows[-1].start, 510)  # short tail is NOT full length

    def test_less_than_window_length(self):
        windows = plan_windows(30, 510, 20)
        self.assertEqual(windows, [Window(0, 30)])

    def test_exact_multiple_of_stride_plus_window(self):
        # length that lands exactly on a stride boundary: no extra tail window
        windows = plan_windows(1000, 510, 20)
        stride = 510 - 20
        starts = list(range(0, 1000 - 510 + 1, stride))
        self.assertEqual(len(windows), len(starts))


class TestPlanWindowsValidation(unittest.TestCase):
    def test_zero_window_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(100, 0, 0)

    def test_negative_window_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(100, -5, 0)

    def test_overlap_equal_to_window_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(100, 64, 64)

    def test_overlap_exceeding_window_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(100, 64, 100)

    def test_negative_overlap_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(100, 64, -1)

    def test_unknown_tail_rule_rejected(self):
        with self.assertRaises(ValueError):
            plan_windows(1000, 510, 20, tail="bogus")  # type: ignore[arg-type]


class TestCoverageInvariant(unittest.TestCase):
    """The covering invariant the plan says to test: 0 to a few thousand, and 100000."""

    def test_coverage_both_tail_rules_many_lengths(self):
        for length in list(range(0, 2001)) + [99999, 100000, 100001]:
            for tail in ("aligned", "short"):
                windows = plan_windows(length, 510, 20, tail=tail)
                self.assertTrue(
                    coverage_ok(windows, length),
                    msg=f"coverage failed at length={length} tail={tail}",
                )

    def test_coverage_small_window_many_overlaps(self):
        for window, overlap in ((64, 0), (64, 20), (125, 20), (128, 64)):
            for length in range(0, 1001):
                windows = plan_windows(length, window, overlap, tail="short")
                self.assertTrue(
                    coverage_ok(windows, length),
                    msg=f"coverage failed at length={length} window={window} overlap={overlap}",
                )


class TestTailRuleCost(unittest.TestCase):
    def test_short_tail_never_exceeds_aligned_tail_in_total_chars(self):
        # short tail is designed to cost less or equal forward chars
        for length in (511, 600, 1000, 1500, 9592):
            aligned = plan_windows(length, 510, 20, tail="aligned")
            short = plan_windows(length, 510, 20, tail="short")
            self.assertLessEqual(total_forward_chars(short), total_forward_chars(aligned))

    def test_same_window_count_both_tail_rules(self):
        for length in (511, 600, 1000, 1500, 9592):
            aligned = plan_windows(length, 510, 20, tail="aligned")
            short = plan_windows(length, 510, 20, tail="short")
            self.assertEqual(len(aligned), len(short))


if __name__ == "__main__":
    unittest.main()
