"""Map cleaned characters to BERT vocabulary ids.

One character maps to exactly one token id (no WordPiece sub-splitting),
so that per-character labels stay aligned with input positions one-to-one.
The mapping is: NFKC-normalise, lower-case, then look up in vocab.txt.
If the NFKC form is not exactly one character, or the single character is
not in the vocabulary, the id for [UNK] is used instead.

PAUSE_LABELS fixes the integer label schema used everywhere else in this
project: index 0 is "no punctuation follows", indices 1-7 are the seven
pause marks in a fixed order (matches punct.text_clean.PAUSE_MARKS).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from punct.text_clean import PAUSE_MARKS

# label 0 = no punctuation; labels 1..7 = the marks in this fixed order
PAUSE_LABELS: List[Optional[str]] = [None] + list(PAUSE_MARKS)
NUM_LABELS = len(PAUSE_LABELS)
_MARK_TO_LABEL: Dict[str, int] = {mark: i for i, mark in enumerate(PAUSE_LABELS) if mark is not None}


def mark_to_label(mark: Optional[str]) -> int:
    """Pause-mark character (or None) -> integer class id in [0, NUM_LABELS)."""
    return 0 if mark is None else _MARK_TO_LABEL[mark]


class CharCodec:
    """Loads vocab.txt once and encodes characters to ids."""

    def __init__(self, vocab_path: str | Path):
        self.vocab: Dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                self.vocab[line.rstrip("\n")] = i
        for special in ("[UNK]", "[CLS]", "[SEP]", "[PAD]"):
            if special not in self.vocab:
                raise ValueError(f"vocab.txt at {vocab_path} is missing {special}")
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_chars_seen = 0  # characters that fell back to [UNK]

    def encode_char(self, ch: str) -> int:
        """One cleaned character -> one vocabulary id."""
        normalised = unicodedata.normalize("NFKC", ch).lower()
        if len(normalised) != 1:
            self.unk_chars_seen += 1
            return self.unk_id
        token_id = self.vocab.get(normalised)
        if token_id is None:
            self.unk_chars_seen += 1
            return self.unk_id
        return token_id

    def encode_chars(self, chars: List[str]) -> List[int]:
        return [self.encode_char(ch) for ch in chars]
