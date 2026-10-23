"""Tests for punct.data_prep, using small synthetic files (no real corpus needed
for the unit tests; a separate integration check against the real corpus is
run manually and logged, not asserted here since it takes tens of seconds)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec
from punct.data_prep import build_splits, check_no_cross_split_leakage, save_split_npz
import numpy as np

VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bert-base-chinese", "vocab.txt")


def _write(path: str, lines) -> None:
    with open(path, "w", encoding="gb18030") as fh:
        for line in lines:
            fh.write(line + "\r\n")


@unittest.skipUnless(os.path.exists(VOCAB_PATH), "bert-base-chinese/vocab.txt not present")
class TestBuildSplits(unittest.TestCase):
    def setUp(self):
        self.codec = CharCodec(VOCAB_PATH)
        self.tmpdir = tempfile.mkdtemp()

    def test_files_route_to_the_right_split_by_month(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"),
               ["19980101-01-001-001 今天 出发 ， 明天 回来 。"])
        _write(os.path.join(self.tmpdir, "1998-05-2003版-带音.txt"),
               ["19980501-01-001-001 今天 出发 ， 明天 回来 。"])
        _write(os.path.join(self.tmpdir, "1998-06-2003版-带音.txt"),
               ["19980601-01-001-001 今天 出发 ， 明天 回来 。"])
        articles, counts, hashes = build_splits(self.tmpdir, self.codec)
        self.assertEqual(len(articles["train"]), 1)
        self.assertEqual(len(articles["val"]), 1)
        self.assertEqual(len(articles["test"]), 1)
        self.assertEqual(len(hashes), 3)

    def test_consecutive_lines_same_article_are_concatenated(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"), [
            "19980101-01-001-001 今天 出发",
            "19980101-01-001-002 明天 回来 。",
        ])
        articles, _, _ = build_splits(self.tmpdir, self.codec)
        self.assertEqual(len(articles["train"]), 1)
        art = articles["train"][0]
        self.assertEqual(len(art.char_ids), len("今天出发明天回来"))

    def test_different_article_numbers_stay_separate(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"), [
            "19980101-01-001-001 今天 出发 。",
            "19980101-01-002-001 明天 回来 。",
        ])
        articles, _, _ = build_splits(self.tmpdir, self.codec)
        self.assertEqual(len(articles["train"]), 2)

    def test_label_at_end_of_article_is_preserved_across_lines(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"), [
            "19980101-01-001-001 今天 出发 ，",
            "19980101-01-001-002 明天 回来 。",
        ])
        articles, _, _ = build_splits(self.tmpdir, self.codec)
        art = articles["train"][0]
        # "发" (index 3) should carry comma's label, "来" (last) the full stop's
        from punct.char_codec import mark_to_label
        self.assertEqual(art.label_ids[3], mark_to_label("，"))
        self.assertEqual(art.label_ids[-1], mark_to_label("。"))

    def test_no_cross_split_leakage_on_distinct_articles(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"),
               ["19980101-01-001-001 今天 出发 。"])
        _write(os.path.join(self.tmpdir, "1998-05-2003版-带音.txt"),
               ["19980501-01-001-001 明天 回来 。"])
        articles, _, _ = build_splits(self.tmpdir, self.codec)
        self.assertEqual(check_no_cross_split_leakage(articles), [])

    def test_empty_article_after_cleaning_is_dropped(self):
        _write(os.path.join(self.tmpdir, "1998-01-2003版-带音.txt"), ["19980101-01-001-001 ＊＊＊"])
        articles, counts, _ = build_splits(self.tmpdir, self.codec)
        self.assertEqual(len(articles["train"]), 0)
        self.assertEqual(counts["train"].as_dict()["empty_after_clean"], 1)


@unittest.skipUnless(os.path.exists(VOCAB_PATH), "bert-base-chinese/vocab.txt not present")
class TestSaveSplitNpz(unittest.TestCase):
    def test_boundaries_sum_matches_total_chars(self):
        codec = CharCodec(VOCAB_PATH)
        tmpdir = tempfile.mkdtemp()
        _write(os.path.join(tmpdir, "1998-01-2003版-带音.txt"), [
            "19980101-01-001-001 今天 出发 。",
            "19980101-01-002-001 明天 回来 。",
        ])
        articles, _, _ = build_splits(tmpdir, codec)
        out_dir = os.path.join(tmpdir, "out")
        save_split_npz("train", articles["train"], out_dir)
        data = np.load(os.path.join(out_dir, "train.npz"), allow_pickle=True)
        self.assertEqual(data["boundaries"][-1], len(data["char_ids"]))
        self.assertEqual(len(data["boundaries"]), len(articles["train"]) + 1)
        self.assertEqual(len(data["char_ids"]), len(data["label_ids"]))


if __name__ == "__main__":
    unittest.main()
