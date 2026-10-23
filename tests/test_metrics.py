"""Tests for punct.metrics — pure numpy, hand-computed small examples."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from punct.metrics import all_none_baseline_accuracy, confusion_matrix, punctuation_prf, sentence_final_f1


class TestConfusionMatrix(unittest.TestCase):
    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0, 1]), np.array([0]), num_labels=8)

    def test_perfect_predictions_diagonal_only(self):
        gold = np.array([0, 1, 2, 0, 1])
        cm = confusion_matrix(gold, gold, num_labels=8)
        self.assertEqual(cm.sum(), 5)
        self.assertEqual(np.trace(cm), 5)

    def test_known_small_confusion(self):
        gold = np.array([1, 1, 2])
        pred = np.array([1, 2, 2])
        cm = confusion_matrix(gold, pred, num_labels=8)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[1, 2], 1)
        self.assertEqual(cm[2, 2], 1)


class TestPunctuationPrf(unittest.TestCase):
    def test_all_none_prediction_gives_zero_micro_f1(self):
        gold = np.array([0, 1, 2, 0, 3])
        pred = np.zeros_like(gold)
        report = punctuation_prf(gold, pred, num_labels=8)
        self.assertEqual(report["micro_f1"], 0.0)

    def test_perfect_prediction_gives_micro_f1_one(self):
        # every one of the 7 punctuation classes (1..7) must appear at least
        # once, otherwise its per-class F1 is correctly 0 (no support) and
        # drags macro_f1 below 1.0 even on an otherwise-perfect prediction
        gold = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        report = punctuation_prf(gold, gold, num_labels=8)
        self.assertEqual(report["micro_f1"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_hand_computed_precision_recall(self):
        # label 1: gold has 3 instances, predicted 1 correctly, 1 wrong class,
        # and predicted label 1 once more where gold was actually label 2
        gold = np.array([1, 1, 1, 2])
        pred = np.array([1, 2, 1, 1])
        report = punctuation_prf(gold, pred, num_labels=8)
        # class 1: tp=2 (positions 0,2), fp=1 (position 3, gold=2 pred=1), fn=1 (position 1, gold=1 pred=2)
        c1 = report["per_class"][1]
        self.assertAlmostEqual(c1["precision"], 2 / 3)
        self.assertAlmostEqual(c1["recall"], 2 / 3)
        self.assertAlmostEqual(c1["f1"], 2 / 3)

    def test_none_label_excluded_from_punct_classes(self):
        gold = np.array([0, 0, 0])
        report = punctuation_prf(gold, gold, num_labels=8)
        self.assertNotIn(0, report["per_class"])

    def test_empty_input_does_not_crash(self):
        gold = np.array([], dtype=np.int64)
        report = punctuation_prf(gold, gold, num_labels=8)
        self.assertEqual(report["micro_f1"], 0.0)
        self.assertEqual(report["n_positions"], 0)


class TestSentenceFinalF1(unittest.TestCase):
    def test_merges_multiple_labels_into_one_positive_class(self):
        # labels 2, 3, 4 are sentence-final (。？！); label 1 (，) is not
        gold = np.array([1, 2, 3, 4, 0])
        pred = np.array([1, 2, 2, 0, 0])  # position 2: gold=3(final) pred=2(final) -> TP
        report = sentence_final_f1(gold, pred, sentence_final_labels=(2, 3, 4))
        # gold positives at indices 1,2,3 (3 total); pred positives at indices 1,2 (2 total)
        # TP: index1 (gold=2,pred=2 both final) and index2 (gold=3,pred=2 both final) = 2
        # FN: index3 (gold=4 final, pred=0 not final) = 1
        # FP: none (both pred-positive positions were also gold-positive)
        self.assertEqual(report["support"], 3)
        self.assertAlmostEqual(report["recall"], 2 / 3)
        self.assertAlmostEqual(report["precision"], 1.0)


class TestAllNoneBaseline(unittest.TestCase):
    def test_matches_manual_fraction(self):
        gold = np.array([0, 0, 0, 1, 2])
        self.assertAlmostEqual(all_none_baseline_accuracy(gold), 3 / 5)

    def test_all_punctuation_gives_zero(self):
        gold = np.array([1, 2, 3])
        self.assertEqual(all_none_baseline_accuracy(gold), 0.0)


if __name__ == "__main__":
    unittest.main()
