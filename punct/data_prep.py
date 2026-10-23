"""Turn the 6 raw corpus files into three monthly splits of continuous article streams.

Split rule (completed-plan.md §2): files named "1998-0{1,2,3,4}-*" go to train,
"1998-05-*" to val, "1998-06-*" to test. Within a file, consecutive lines whose
record id shares the same first 3 fields (YYYYMMDD-PP-AAA) belong to one
article; an article's cleaned lines are concatenated in file order into one
continuous character stream, with one label stream of the same length.

Output per split: a list of Article(chars, labels, source_file) plus the
split's CleanCounts, written to disk as an .npz (ids + labels + per-article
boundary offsets, so no article information is lost) and a data_audit.json
with file hashes and the cleaning counters for every split.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from punct.char_codec import CharCodec, mark_to_label
from punct.text_clean import CleanCounts, clean_line

_MONTH_RE = re.compile(r"1998-(\d\d)")
_ARTICLE_KEY_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})")
SPLIT_OF_MONTH = {"01": "train", "02": "train", "03": "train", "04": "train", "05": "val", "06": "test"}
SPLITS = ("train", "val", "test")


@dataclass
class Article:
    source_file: str
    article_key: str
    char_ids: List[int]
    label_ids: List[int]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_to_split(path: str) -> str:
    m = _MONTH_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot determine month from file name: {path}")
    return SPLIT_OF_MONTH[m.group(1)]


def build_splits(corpus_dir: str, codec: CharCodec) -> Tuple[Dict[str, List[Article]], Dict[str, CleanCounts], Dict[str, str]]:
    """Read all 6 files once, return (articles_by_split, counts_by_split, file_hashes)."""
    articles: Dict[str, List[Article]] = {s: [] for s in SPLITS}
    counts: Dict[str, CleanCounts] = {s: CleanCounts() for s in SPLITS}
    file_hashes: Dict[str, str] = {}

    # per-file accumulation: article_key -> (chars, labels), in first-seen order
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        split = _file_to_split(path)
        file_hashes[os.path.basename(path)] = sha256_of(path)
        order: List[str] = []
        pending: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # key -> [(char_id, label_id), ...]
        with open(path, "r", encoding="gb18030", errors="replace") as fh:
            for raw_line in fh:
                raw_line = raw_line.rstrip("\r\n")
                if not raw_line.strip():
                    continue
                result = clean_line(raw_line)
                counts[split].update(raw_line, result)
                key_match = _ARTICLE_KEY_RE.match(raw_line)
                key = key_match.group(1) if key_match else f"__no_id__:{len(order)}"
                if key not in pending:
                    order.append(key)
                char_ids = codec.encode_chars(result.chars)
                label_ids = [mark_to_label(m) for m in result.label_marks]
                pending[key].extend(zip(char_ids, label_ids))
        for key in order:
            pairs = pending[key]
            if not pairs:
                continue
            chars, labels = zip(*pairs)
            articles[split].append(
                Article(source_file=os.path.basename(path), article_key=key,
                        char_ids=list(chars), label_ids=list(labels))
            )
    return articles, counts, file_hashes


def check_no_cross_split_leakage(articles: Dict[str, List[Article]]) -> List[str]:
    """Return a list of article_keys that appear in more than one split (should be empty)."""
    key_to_splits: Dict[str, set] = defaultdict(set)
    for split, arts in articles.items():
        for a in arts:
            key_to_splits[a.article_key].add(split)
    return [k for k, splits in key_to_splits.items() if len(splits) > 1]


def save_split_npz(split: str, arts: List[Article], out_dir: str) -> None:
    """Flatten one split's articles into (char_ids, label_ids, article boundaries)."""
    os.makedirs(out_dir, exist_ok=True)
    char_ids: List[int] = []
    label_ids: List[int] = []
    boundaries = [0]
    source_files: List[str] = []
    article_keys: List[str] = []
    for a in arts:
        char_ids.extend(a.char_ids)
        label_ids.extend(a.label_ids)
        boundaries.append(len(char_ids))
        source_files.append(a.source_file)
        article_keys.append(a.article_key)
    np.savez_compressed(
        os.path.join(out_dir, f"{split}.npz"),
        char_ids=np.asarray(char_ids, dtype=np.int32),
        label_ids=np.asarray(label_ids, dtype=np.int8),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        source_files=np.asarray(source_files, dtype=object),
        article_keys=np.asarray(article_keys, dtype=object),
    )


def write_audit(
    articles: Dict[str, List[Article]],
    counts: Dict[str, CleanCounts],
    file_hashes: Dict[str, str],
    leaked_keys: List[str],
    codec: CharCodec,
    out_path: str,
) -> None:
    audit = {
        "file_sha256": file_hashes,
        "vocab_unk_chars_seen": codec.unk_chars_seen,
        "cross_split_leaked_article_keys": leaked_keys,
        "splits": {},
    }
    for split in SPLITS:
        arts = articles[split]
        n_chars = sum(len(a.char_ids) for a in arts)
        audit["splits"][split] = {
            "articles": len(arts),
            "chars": n_chars,
            "clean_counts": counts[split].as_dict(),
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, ensure_ascii=False, indent=2)


def main(corpus_dir: str, vocab_path: str, data_dir: str, audit_path: str) -> None:
    codec = CharCodec(vocab_path)
    articles, counts, file_hashes = build_splits(corpus_dir, codec)
    leaked = check_no_cross_split_leakage(articles)
    for split in SPLITS:
        save_split_npz(split, articles[split], data_dir)
    write_audit(articles, counts, file_hashes, leaked, codec, audit_path)
    for split in SPLITS:
        n_chars = sum(len(a.char_ids) for a in articles[split])
        print(f"{split}: articles={len(articles[split])} chars={n_chars}")
    print(f"cross-split leaked article keys: {len(leaked)}")
    print(f"UNK fallbacks during encoding: {codec.unk_chars_seen}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
