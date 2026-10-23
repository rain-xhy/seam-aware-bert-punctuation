"""Map cleaned characters to BERT vocabulary ids.

One character maps to exactly one token id (no WordPiece sub-splitting),
so that per-character labels stay aligned with input positions one-to-one.
The mapping is: NFKC-normalise, lower-case, then look up in vocab.txt.
If the NFKC form is not exactly one character, or the single character is
not in the vocabulary, the id for [UNK] is used instead.

IMPORTANT: vocab.txt line number == token id. Lines must never be filtered,
stripped to nothing, or skipped (vocab.txt contains a line holding U+2028 that
looks blank after str.strip(); dropping it shifts every later id by one and
silently breaks the pretrained embeddings). Only the trailing newline is removed.

PAUSE_LABELS fixes the integer label schema used everywhere else in this
project: index 0 is "no punctuation follows", indices 1-7 are the seven
pause marks in a fixed order (matches punct.text_clean.PAUSE_MARKS).
"""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from punct.text_clean import PAUSE_MARKS

# label 0 = no punctuation; labels 1..7 = the marks in this fixed order
PAUSE_LABELS: List[Optional[str]] = [None] + list(PAUSE_MARKS)
NUM_LABELS = len(PAUSE_LABELS)
_MARK_TO_LABEL: Dict[str, int] = {mark: i for i, mark in enumerate(PAUSE_LABELS) if mark is not None}

UNK_TOKEN = "[UNK]"


def mark_to_label(mark: Optional[str]) -> int:
    """Pause-mark character (or None) -> integer class id in [0, NUM_LABELS)."""
    return 0 if mark is None else _MARK_TO_LABEL[mark]


def read_vocab_lines(vocab_path: Union[str, Path]) -> List[str]:
    """Read vocab.txt keeping line number == id (only the line terminator is removed)."""
    vocab: List[str] = []
    with open(vocab_path, "r", encoding="utf-8", newline="") as fh:
        data = fh.read()
    # split on "\n" only; str.splitlines() would also split on U+2028 and shift ids
    lines = data.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        vocab.append(line[:-1] if line.endswith("\r") else line)
    return vocab


class CharCodec:
    """Character -> vocabulary id. Accepts a vocab.txt path or a list of tokens."""

    def __init__(self, vocab: Union[str, Path, Sequence[str]]):
        if isinstance(vocab, (str, Path)):
            tokens = read_vocab_lines(vocab)
            src = str(vocab)
        else:
            tokens = list(vocab)
            src = "<list>"
        self.vocab_list: List[str] = tokens
        self.vocab: Dict[str, int] = {}
        for i, tok in enumerate(tokens):
            self.vocab.setdefault(tok, i)
        self.id_to_char: Dict[int, str] = {i: t for i, t in enumerate(tokens)}
        for special in ("[UNK]", "[CLS]", "[SEP]", "[PAD]"):
            if special not in self.vocab:
                raise ValueError(f"vocab ({src}) is missing {special}")
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_chars_seen = 0  # characters that fell back to [UNK]

    # ---- encoding -------------------------------------------------
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

    def encode_chars(self, chars: Sequence[str]) -> List[int]:
        return [self.encode_char(ch) for ch in chars]

    def encode(self, text: Sequence[str]) -> List[int]:
        return self.encode_chars(list(text))

    # ---- decoding -------------------------------------------------
    def decode_char(self, idx: int) -> str:
        return self.id_to_char.get(int(idx), UNK_TOKEN)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.decode_char(i) for i in ids)

    # ---- persistence ---------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab_list, "vocab_size": len(self.vocab_list)},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharCodec":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"])

    @classmethod
    def from_bert_vocab(cls, bert_dir: Union[str, Path]) -> "CharCodec":
        vocab_path = os.path.join(str(bert_dir), "vocab.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.txt not found in {bert_dir}")
        codec = cls(vocab_path)
        return codec
