import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.convert_clean_data import TRAIN_ORDER, convert_split, label_remap  # noqa: E402
from punct.char_codec import PAUSE_LABELS  # noqa: E402


class ConvertTests(unittest.TestCase):
    def test_remap_preserves_mark_identity(self):
        remap = label_remap()
        for old, mark in enumerate(PAUSE_LABELS):
            self.assertEqual(TRAIN_ORDER[remap[old]], mark)

    def test_remap_is_permutation_fixing_none(self):
        remap = label_remap()
        self.assertEqual(sorted(remap.tolist()), list(range(8)))
        self.assertEqual(remap[0], 0)

    def test_sentence_final_ids_are_2_3_4(self):
        remap = label_remap()
        marks = {TRAIN_ORDER[i] for i in (2, 3, 4)}
        self.assertEqual(marks, {"。", "？", "！"})
        for old, mark in enumerate(PAUSE_LABELS):
            if mark in marks:
                self.assertIn(int(remap[old]), (2, 3, 4))

    def test_convert_split_splits_articles(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "x.npz")
            old = {m: i for i, m in enumerate(PAUSE_LABELS) if m}
            labels = np.array([0, old["、"], 0, old["？"], 0], dtype=np.int8)
            np.savez(p, char_ids=np.arange(5, dtype=np.int32), label_ids=labels,
                     boundaries=np.array([0, 2, 5], dtype=np.int64))
            chars, labs = convert_split(p, label_remap())
        self.assertEqual([len(c) for c in chars], [2, 3])
        self.assertEqual(int(labs[0][1]), TRAIN_ORDER.index("、"))
        self.assertEqual(int(labs[1][0]), 0)
        self.assertEqual(int(labs[1][1]), TRAIN_ORDER.index("？"))


if __name__ == "__main__":
    unittest.main()
