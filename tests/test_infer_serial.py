"""Tests for punct.infer_serial, using a deterministic fake model (position-
based, same technique as test_merge.py) so the decoding LOGIC is tested
independently of BERT quality."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from punct.infer_serial import (
    cross_article_serial,
    tang_serial_v1,
    tang_serial_v2,
    teacher_serial,
)

NONE, COMMA, PERIOD, QUESTION, EXCLAIM = 0, 1, 2, 3, 4
SENTENCE_FINAL = (PERIOD, QUESTION, EXCLAIM)


def make_fake_model(label_at: dict):
    """Returns a model_fn(chunk) -> labels, where label_at maps a GLOBAL
    char id (the fake 'character', here just an int) to its label. Any char
    id not in label_at gets NONE. Since the chunk IS the list of global
    ids in these tests (see make_article below), this directly controls
    where sentence-final marks land."""

    def model_fn(chunk):
        return np.array([label_at.get(c, NONE) for c in chunk], dtype=np.int64)

    return model_fn


def make_article_with_periods_every(n: int, period_every: int):
    """char_ids are just 0..n-1; periods (PERIOD label) land every
    `period_every` positions starting at period_every-1."""
    char_ids = list(range(n))
    label_at = {i: PERIOD for i in range(period_every - 1, n, period_every)}
    return char_ids, label_at


class TestTeacherSerial(unittest.TestCase):
    def test_short_article_single_window_commits_everything(self):
        char_ids, label_at = make_article_with_periods_every(50, 100)  # no period inside 50 chars
        model_fn = make_fake_model(label_at)
        labels, calls = teacher_serial(char_ids, model_fn, window=400, commit_prefix=300)
        self.assertEqual(calls, 1)
        self.assertEqual(len(labels), 50)

    def test_commits_up_to_last_period_within_commit_prefix(self):
        # periods every 50 chars; window=400, commit_prefix=300 -> expect a
        # period at position 249 (0-indexed) to be the cut point in the
        # first window, i.e. the labels up through 249 should be committed
        # from THIS window's own prediction
        char_ids, label_at = make_article_with_periods_every(1000, 50)
        model_fn = make_fake_model(label_at)
        labels, calls = teacher_serial(char_ids, model_fn, window=400, commit_prefix=300)
        self.assertEqual(labels[249], PERIOD)
        # next window should start at 250
        self.assertGreater(calls, 1)

    def test_no_period_in_commit_prefix_falls_back_to_committing_prefix_length(self):
        char_ids, label_at = make_article_with_periods_every(1000, 1000)  # period only at the very end, outside window
        model_fn = make_fake_model(label_at)
        labels, calls = teacher_serial(char_ids, model_fn, window=400, commit_prefix=300)
        # first window [0,400) has no period in labels[:300], so it commits
        # exactly 300 chars and starts the next window at 300
        self.assertGreaterEqual(calls, 2)

    def test_every_position_gets_a_label_eventually(self):
        char_ids, label_at = make_article_with_periods_every(2000, 37)
        model_fn = make_fake_model(label_at)
        labels, _ = teacher_serial(char_ids, model_fn, window=400, commit_prefix=300)
        self.assertEqual(len(labels), 2000)
        # every period position that teacher_serial could see should be committed
        for i in label_at:
            if label_at[i] == PERIOD:
                self.assertIn(labels[i], (PERIOD, NONE))  # NONE only possible if beyond model's reach, should not happen here
        self.assertEqual(sum(1 for i, lab in enumerate(labels) if lab == PERIOD),
                          sum(1 for lab in label_at.values() if lab == PERIOD))


class TestTangSerialV1(unittest.TestCase):
    def test_keeps_only_first_two_sentences(self):
        # window=64, periods at 10, 20, 30, 40, 50, 60 within the window
        char_ids = list(range(200))
        label_at = {i: PERIOD for i in (9, 19, 29, 39, 49, 59)}
        model_fn = make_fake_model(label_at)
        labels, calls = tang_serial_v1(char_ids, model_fn, window=64, keep_sentences=2)
        # first window should commit through position 19 (2nd period) only
        self.assertEqual(labels[19], PERIOD)
        self.assertGreater(calls, 1)

    def test_fewer_periods_than_keep_sentences_commits_what_exists(self):
        char_ids = list(range(200))
        label_at = {9: PERIOD}  # only one period in the first window
        model_fn = make_fake_model(label_at)
        labels, calls = tang_serial_v1(char_ids, model_fn, window=64, keep_sentences=2)
        self.assertEqual(labels[9], PERIOD)

    def test_short_article_commits_everything(self):
        char_ids, label_at = make_article_with_periods_every(30, 100)
        model_fn = make_fake_model(label_at)
        labels, calls = tang_serial_v1(char_ids, model_fn, window=64)
        self.assertEqual(calls, 1)
        self.assertEqual(len(labels), 30)


class TestTangSerialV2(unittest.TestCase):
    def test_drops_only_last_sentence(self):
        # window=125, periods at 30, 60, 90, 120 -> should drop only the
        # last one (120), committing through 90
        char_ids = list(range(500))
        label_at = {i: PERIOD for i in (29, 59, 89, 119)}
        model_fn = make_fake_model(label_at)
        labels, calls = tang_serial_v2(char_ids, model_fn, window=125)
        self.assertEqual(labels[89], PERIOD)
        self.assertGreater(calls, 1)

    def test_no_period_commits_whole_window(self):
        char_ids, label_at = make_article_with_periods_every(1000, 1000)
        model_fn = make_fake_model(label_at)
        labels, calls = tang_serial_v2(char_ids, model_fn, window=125)
        self.assertGreaterEqual(calls, 1000 // 125)

    def test_every_position_eventually_labelled(self):
        char_ids, label_at = make_article_with_periods_every(600, 41)
        model_fn = make_fake_model(label_at)
        labels, _ = tang_serial_v2(char_ids, model_fn, window=125)
        self.assertEqual(len(labels), 600)


class TestCrossArticleSerial(unittest.TestCase):
    def test_batches_multiple_articles_per_call(self):
        articles = [list(range(1000)), list(range(1000, 1600))]  # lengths 1000, 600

        def batch_model_fn(chunks):
            return [np.zeros(len(c), dtype=np.int64) for c in chunks]  # all NONE

        labels, calls = cross_article_serial(articles, batch_model_fn, window=400, commit_prefix=300)
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(labels[0]), 1000)
        self.assertEqual(len(labels[1]), 600)
        # with no sentence-final marks anywhere, each article commits 300
        # chars per step until its tail: article 0 needs ceil-ish 3 steps,
        # article 1 needs 2 steps, but they run IN LOCKSTEP so total calls
        # should be close to max(steps needed), much less than sum
        self.assertLess(calls, 10)

    def test_shorter_articles_finish_early_and_drop_out_of_batch(self):
        articles = [list(range(100)), list(range(100, 1100))]  # lengths 100, 1000
        call_sizes = []

        def batch_model_fn(chunks):
            call_sizes.append(len(chunks))
            return [np.zeros(len(c), dtype=np.int64) for c in chunks]

        cross_article_serial(articles, batch_model_fn, window=400, commit_prefix=300)
        self.assertEqual(call_sizes[0], 2)  # both articles active on step 1
        self.assertLess(call_sizes[-1], 2)  # the short article finished before the long one

    def test_all_positions_labelled(self):
        articles = [list(range(737))]
        label_at = {i: PERIOD for i in (100, 250, 400, 600)}

        def batch_model_fn(chunks):
            return [np.array([label_at.get(c, NONE) for c in chunk], dtype=np.int64) for chunk in chunks]

        labels, _ = cross_article_serial(articles, batch_model_fn, window=400, commit_prefix=300)
        self.assertEqual(len(labels[0]), 737)


if __name__ == "__main__":
    unittest.main()
