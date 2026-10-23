"""Prepare training data from raw corpus (per-article storage).

Reads 1998-01-2003版-带音.txt (GB18030), groups lines into articles by their
prefixes (19980101-01-001 = one article), cleans (remove word-spacing, pinyin
braces {shang5}, non-target punctuation), encodes chars and labels, and saves
per-article numpy arrays for random cropping during training.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Target punctuation labels (0=O, 1-7=punctuation)
LABELS = ['O', ',', '。', '?', '!', ';', ':', '、']
LABEL_TO_ID = {ch: i for i, ch in enumerate(LABELS)}


def parse_article_id(line: str) -> str:
    """Extract article ID from line prefix.
    19980101-01-001-002 -> 19980101-01-001"""
    match = re.match(r'^(\d{8}-\d{2}-\d{3})', line)
    return match.group(1) if match else None


def clean_line(line: str) -> str:
    """Remove line prefix, pinyin braces, and word-spacing."""
    # Remove prefix (19980101-01-001-002)
    line_no_prefix = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}\s+', '', line)
    # Remove pinyin braces
    line_no_pinyin = re.sub(r'\{[^}]*\}', '', line_no_prefix)
    # Remove word-spacing
    line_no_space = re.sub(r'[ \t]+', '', line_no_pinyin)
    return line_no_space


def extract_chars_labels(text: str) -> Tuple[List[str], List[int]]:
    """Extract characters and labels from cleaned text.

    Returns:
        chars: list of characters (punctuation removed)
        labels: label IDs, one per character (0=O, 1-7=punctuation after this char)
    """
    chars = []
    labels = []
    for ch in text:
        if ch in LABEL_TO_ID:
            # Target punctuation: label on previous char
            if chars:
                labels[-1] = LABEL_TO_ID[ch]
        elif ch in '\r\n':
            # Skip newlines
            continue
        else:
            # Regular character
            chars.append(ch)
            labels.append(0)  # default: no punctuation
    return chars, labels


def group_by_article(corpus_path: str) -> Dict[str, str]:
    """Read corpus and group lines into articles.

    Returns:
        {article_id: article_text}
    """
    with open(corpus_path, "r", encoding="gb18030", errors="replace") as f:
        lines = f.readlines()

    articles = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        article_id = parse_article_id(line)
        if article_id:
            cleaned = clean_line(line)
            articles[article_id].append(cleaned)

    # Join lines within each article
    return {aid: ''.join(lines) for aid, lines in articles.items()}


def encode_articles(articles: Dict[str, str], codec) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Encode articles to char IDs and label IDs.

    Returns:
        articles_char: list of numpy arrays (char IDs)
        articles_label: list of numpy arrays (label IDs)
    """
    articles_char = []
    articles_label = []

    for article_id in sorted(articles.keys()):
        text = articles[article_id]
        chars, labels = extract_chars_labels(text)

        # Encode chars to IDs
        char_ids = np.array([codec.encode_char(ch) for ch in chars], dtype=np.int32)
        label_ids = np.array(labels, dtype=np.int32)

        # Skip very short articles
        if len(char_ids) < 32:
            continue

        articles_char.append(char_ids)
        articles_label.append(label_ids)

    return articles_char, articles_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="../1998-上半年-带音/1998-01-2003版-带音.txt")
    parser.add_argument("--bert_dir", default="bert-base-chinese")
    parser.add_argument("--out_dir", default="prepared_data")
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    args = parser.parse_args()

    # Read and group by article
    print(f"Reading {args.corpus} ...")
    articles = group_by_article(args.corpus)
    print(f"  {len(articles)} articles")

    # Build codec
    print(f"Loading vocab from {args.bert_dir}/vocab.txt ...")
    from punct.char_codec import CharCodec
    codec = CharCodec.from_bert_vocab(args.bert_dir)

    # Encode articles
    print("Encoding articles...")
    articles_char, articles_label = encode_articles(articles, codec)
    print(f"  {len(articles_char)} articles after filtering")

    total_chars = sum(len(a) for a in articles_char)
    total_puncts = sum(np.sum(a > 0) for a in articles_label)
    print(f"  {total_chars} chars, {total_puncts} punctuation marks")

    # Train/dev split
    split_idx = int(len(articles_char) * (1 - args.dev_ratio))
    train_char = articles_char[:split_idx]
    train_label = articles_label[:split_idx]
    dev_char = articles_char[split_idx:]
    dev_label = articles_label[split_idx:]
    print(f"  train: {len(train_char)} articles, dev: {len(dev_char)} articles")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "train_articles_char.npy"), np.array(train_char, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.out_dir, "train_articles_label.npy"), np.array(train_label, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.out_dir, "dev_articles_char.npy"), np.array(dev_char, dtype=object), allow_pickle=True)
    np.save(os.path.join(args.out_dir, "dev_articles_label.npy"), np.array(dev_label, dtype=object), allow_pickle=True)

    codec_path = os.path.join(args.out_dir, "char_codec.json")
    codec.save(codec_path)

    print(f"Saved to {args.out_dir}/")
    print(f"  train_articles_char.npy: {len(train_char)} articles")
    print(f"  dev_articles_char.npy: {len(dev_char)} articles")
    print(f"  char_codec.json: {len(codec.vocab)} vocab entries")


if __name__ == "__main__":
    main()
