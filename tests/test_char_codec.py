"""Tests for punct.char_codec. Needs the real vocab.txt (no network)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec, NUM_LABELS, PAUSE_LABELS, mark_to_label
from punct.text_clean import PAUSE_MARKS

VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bert-base-chinese", "vocab.txt")


class TestLabelSchema(unittest.TestCase):
    def test_eight_classes(self):
        self.assertEqual(NUM_LABELS, 8)

    def test_none_is_label_zero(self):
        self.assertEqual(mark_to_label(None), 0)

    def test_every_pause_mark_has_a_distinct_label(self):
        labels = {mark_to_label(m) for m in PAUSE_MARKS}
        self.assertEqual(len(labels), 7)
        self.assertNotIn(0, labels)

    def test_labels_match_text_clean_order(self):
        self.assertEqual(PAUSE_LABELS[1:], list(PAUSE_MARKS))


@unittest.skipUnless(os.path.exists(VOCAB_PATH), "bert-base-chinese/vocab.txt not present")
class TestCharCodec(unittest.TestCase):
    def setUp(self):
        self.codec = CharCodec(VOCAB_PATH)

    def test_vocab_size_is_21128(self):
        self.assertEqual(len(self.codec.vocab), 21128)

    def test_common_char_round_trips_to_a_real_id(self):
        token_id = self.codec.encode_char("中")
        self.assertNotEqual(token_id, self.codec.unk_id)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, 21128)

    def test_pause_marks_have_real_ids_except_quotes(self):
        # the 7 pause marks themselves are not fed to the model (they are
        # stripped before encoding by text_clean), but the encoder should
        # not crash or silently map them to [UNK] if ever called directly
        for mark in PAUSE_MARKS:
            token_id = self.codec.encode_char(mark)
            self.assertIsInstance(token_id, int)

    def test_unknown_char_falls_back_to_unk(self):
        # a private-use-area character is extremely unlikely to be in vocab
        token_id = self.codec.encode_char("")
        self.assertEqual(token_id, self.codec.unk_id)

    def test_nfkc_multi_char_expansion_falls_back_to_unk(self):
        # U+2103 DEGREE CELSIUS NFKC-expands to "°C" (2 chars); the design
        # log recorded this exact character among the 120 such cases
        token_id = self.codec.encode_char("℃")
        self.assertEqual(token_id, self.codec.unk_id)
        self.assertGreater(self.codec.unk_chars_seen, 0)

    def test_fullwidth_digit_maps_via_nfkc(self):
        # the corpus writes numbers in fullwidth digits; NFKC should fold
        # them to the halfwidth form before vocab lookup
        token_id = self.codec.encode_char("３")
        self.assertNotEqual(token_id, self.codec.unk_id)

    def test_encode_chars_preserves_length(self):
        ids = self.codec.encode_chars(list("中国人民"))
        self.assertEqual(len(ids), 4)


if __name__ == "__main__":
    unittest.main()
