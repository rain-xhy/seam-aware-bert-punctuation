"""Clean one raw corpus line into (chars, labels, counts).

Rules (same as the exploratory probes in ../*_probe.py, now consolidated):
  - strip the leading "YYYYMMDD-PP-AAA-NNN" record id
  - strip pinyin annotations written as {xxxx}
  - strip inter-word spaces (ASCII and fullwidth U+3000)
  - of the 7 pause marks ，。、；：？！ : the mark labels the character
    before it; if two pause marks land on the same character (because
    something between them, e.g. a quote, was dropped), the later one wins
    and the conflict is counted
  - every other Unicode category "P" character is dropped from the input,
    EXCEPT a fullwidth full stop between two ASCII digits, which is kept as
    an ordinary input character (decimal point, e.g. "3．5")
  - remaining characters are NFKC-normalised and lower-cased before vocab
    lookup; characters whose NFKC form is not exactly one character are
    flagged (counted) but still looked up via the first NFKC character
    mapped through the normal path in char_codec (not handled here)

This module only turns one line into (chars, label_marks, counts). Turning
label_marks (the literal punctuation character or None) into integer class
ids is done by the caller, so this module has no notion of a label schema.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

_ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
_PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
_SPACES = (" ", "　")

PAUSE_MARKS = "，。、；：？！"
_DECIMAL_DOT = "．"


@dataclass
class CleanResult:
    """Output of cleaning one line."""

    article_id: Optional[str]
    chars: List[str] = field(default_factory=list)
    # label_marks[i] is the pause-mark character that follows chars[i], or
    # None if no pause mark follows it.
    label_marks: List[Optional[str]] = field(default_factory=list)


def clean_line(raw_line: str) -> CleanResult:
    """Clean a single raw corpus line.

    Returns the article id (from the leading record id, or None if the line
    has no such id) together with the kept characters and, for each kept
    character, the pause mark that immediately follows it (if any).
    """
    match = _ID_RE.match(raw_line)
    article_id = match.group(1) if match else None
    body = _ID_RE.sub("", raw_line, count=1)
    body = _PINYIN_RE.sub("", body)
    for sp in _SPACES:
        body = body.replace(sp, "")

    chars: List[str] = []
    marks: List[Optional[str]] = []
    for i, ch in enumerate(body):
        if ch in PAUSE_MARKS:
            if marks:
                # Two pause marks landed on the same character: the later
                # one overwrites, per the clean_counts conflict rule.
                marks[-1] = ch
            continue
        category = unicodedata.category(ch)
        if category.startswith("P"):
            if ch == _DECIMAL_DOT and _is_decimal_point(body, i):
                chars.append(ch)
                marks.append(None)
            # every other punctuation / quote / bracket character: dropped,
            # not kept as input and not a label target
            continue
        chars.append(ch)
        marks.append(None)

    return CleanResult(article_id=article_id, chars=chars, label_marks=marks)


def _is_decimal_point(body: str, pos: int) -> bool:
    """True if body[pos] (a fullwidth full stop) sits between two digit characters.

    The corpus writes numbers in fullwidth digits (e.g. "３．５", not "3.5");
    a probe over the full corpus found zero halfwidth digits next to this
    dot. str.isdigit() is true for both widths, so neither side is further
    restricted to ASCII.
    """
    left = body[pos - 1] if pos > 0 else ""
    right = body[pos + 1] if pos + 1 < len(body) else ""
    return left.isdigit() and right.isdigit()


@dataclass
class CleanCounts:
    """Running totals across many cleaned lines, for the data audit log."""

    lines: int = 0
    empty_after_clean: int = 0
    pause_conflicts: int = 0
    decimal_points_kept: int = 0
    other_punct_dropped: Counter = field(default_factory=Counter)

    def update(self, raw_line: str, result: CleanResult) -> None:
        self.lines += 1
        if not result.chars:
            self.empty_after_clean += 1
        body = _PINYIN_RE.sub("", _ID_RE.sub("", raw_line, count=1))
        for sp in _SPACES:
            body = body.replace(sp, "")
        pending_pause = False
        for i, ch in enumerate(body):
            if ch in PAUSE_MARKS:
                if pending_pause:
                    self.pause_conflicts += 1
                pending_pause = True
                continue
            pending_pause = False
            if unicodedata.category(ch).startswith("P"):
                if ch == _DECIMAL_DOT and _is_decimal_point(body, i):
                    self.decimal_points_kept += 1
                else:
                    self.other_punct_dropped[ch] += 1

    def as_dict(self) -> dict:
        return {
            "lines": self.lines,
            "empty_after_clean": self.empty_after_clean,
            "pause_conflicts": self.pause_conflicts,
            "decimal_points_kept": self.decimal_points_kept,
            "other_punct_dropped_total": sum(self.other_punct_dropped.values()),
            "other_punct_dropped_top20": self.other_punct_dropped.most_common(20),
        }
