"""Tests for punct.text_clean, written before the module is trusted with real data.

Run: python -m pytest tests/test_text_clean.py -q
(or python -m unittest tests.test_text_clean -v if pytest is not installed)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.text_clean import CleanCounts, PAUSE_MARKS, clean_line


class TestCleanLine(unittest.TestCase):
    def test_strips_record_id_and_spaces(self):
        r = clean_line("19980101-01-001-001 今天 出发 ， 明天 回来 。")
        self.assertEqual(r.article_id, "19980101-01-001")
        self.assertEqual("".join(r.chars), "今天出发明天回来")

    def test_label_on_preceding_char(self):
        r = clean_line("19980101-01-001-001 今天 出发 ， 明天 回来 。")
        # "发" is the 4th char, followed by the comma
        self.assertEqual(r.chars[3], "发")
        self.assertEqual(r.label_marks[3], "，")
        self.assertEqual(r.chars[-1], "来")
        self.assertEqual(r.label_marks[-1], "。")
        # no other position has a label
        self.assertEqual(sum(1 for m in r.label_marks if m is not None), 2)

    def test_pinyin_annotation_removed(self):
        r = clean_line("19980101-01-001-007 这{zhe4} 一 年")
        self.assertEqual("".join(r.chars), "这一年")

    def test_no_record_id_gives_none_article(self):
        r = clean_line("no id here at all")
        self.assertIsNone(r.article_id)

    def test_empty_line_after_cleaning(self):
        r = clean_line("19980101-01-001-050 ＊＊＊")
        self.assertEqual(r.chars, [])
        self.assertEqual(r.label_marks, [])

    def test_decimal_point_between_fullwidth_digits_is_kept(self):
        # the real corpus writes numbers in fullwidth digits, e.g. "３．５"
        # (confirmed: a full-corpus scan found zero halfwidth digits next
        # to this dot), so that is what this test exercises
        r = clean_line("19980101-01-001-001 比价 为 ３．５ 元")
        text = "".join(r.chars)
        self.assertIn("．", text)

    def test_decimal_point_between_halfwidth_digits_is_also_kept(self):
        # not observed in the corpus, but the rule is width-agnostic
        r = clean_line("19980101-01-001-001 比价 为 3．5 元")
        text = "".join(r.chars)
        self.assertIn("．", text)

    def test_decimal_point_not_between_digits_is_dropped(self):
        # a fullwidth full stop used as an actual sentence end, with a
        # non-digit on at least one side, is treated as a normal pause mark
        # position, not a kept input character
        r = clean_line("19980101-01-001-001 全文 结束 ．")
        self.assertNotIn("．", "".join(r.chars))

    def test_other_punctuation_dropped_not_kept_as_input(self):
        r = clean_line("19980101-01-001-001 他 说 “ 你好 ” 。")
        text = "".join(r.chars)
        self.assertNotIn("“", text)
        self.assertNotIn("”", text)
        # the char right before the closing quote should NOT get a label
        # from the quote itself (quotes are not pause marks)
        self.assertNotIn("“", PAUSE_MARKS)

    def test_adjacent_pause_marks_conflict_keeps_later_one(self):
        # simulate what happens once a quote between two pause marks is
        # dropped: the comma is immediately followed by a full stop on the
        # same preceding character
        r = clean_line("19980101-01-001-001 结束 ， 。")
        self.assertEqual(r.chars[-1], "束")
        self.assertEqual(r.label_marks[-1], "。")  # later mark wins

    def test_all_seven_pause_marks_recognised(self):
        for mark in PAUSE_MARKS:
            r = clean_line(f"19980101-01-001-001 字{mark}")
            self.assertEqual(r.chars, ["字"], msg=f"mark={mark!r}")
            self.assertEqual(r.label_marks, [mark], msg=f"mark={mark!r}")


class TestCleanCounts(unittest.TestCase):
    def test_counts_match_known_bad_bytes_log(self):
        # bad_bytes_probe.log recorded 5 undecodable bytes total across the
        # corpus; this test only checks the counter mechanics, not that log.
        counts = CleanCounts()
        counts.update("19980101-01-001-001 今天 ， 。", clean_line("19980101-01-001-001 今天 ， 。"))
        d = counts.as_dict()
        self.assertEqual(d["lines"], 1)
        self.assertEqual(d["empty_after_clean"], 0)

    def test_empty_after_clean_counted(self):
        counts = CleanCounts()
        line = "19980101-01-001-050 ＊＊＊"
        counts.update(line, clean_line(line))
        self.assertEqual(counts.as_dict()["empty_after_clean"], 1)

    def test_pause_conflict_counted(self):
        counts = CleanCounts()
        line = "19980101-01-001-001 结束 ， 。"
        counts.update(line, clean_line(line))
        self.assertEqual(counts.as_dict()["pause_conflicts"], 1)

    def test_decimal_point_counted(self):
        counts = CleanCounts()
        line = "19980101-01-001-001 比价 为 ３．５ 元"  # fullwidth digits, as in the corpus
        counts.update(line, clean_line(line))
        self.assertEqual(counts.as_dict()["decimal_points_kept"], 1)

    def test_other_punct_dropped_counted(self):
        counts = CleanCounts()
        line = "19980101-01-001-001 他 说 “ 你好 ” 。"
        counts.update(line, clean_line(line))
        d = counts.as_dict()
        self.assertGreaterEqual(d["other_punct_dropped_total"], 2)  # the two quotes


if __name__ == "__main__":
    unittest.main()
