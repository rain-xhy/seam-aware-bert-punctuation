#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare complete 1-6 month data with proper train/dev/test split."""
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

CORPUS_DIR = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\1998-上半年-带音")
OUT_DIR = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\prepared_data_complete")
OUT_DIR.mkdir(exist_ok=True)

LABELS = ['O', '，', '。', '？', '！', '；', '：', '、']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def parse_article_id(line):
    """Extract article ID: 19980101-01-001-002 -> 19980101-01-001"""
    match = re.match(r'^(\d{8}-\d{2}-\d{3})', line)
    return match.group(1) if match else None

def clean_line(line):
    """Remove line prefix, pinyin, word-spacing."""
    line = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}\s+', '', line)
    line = re.sub(r'\{[^}]*\}', '', line)
    line = re.sub(r'[ \t]+', '', line)
    return line

def extract_chars_labels(text):
    """Extract (char, label) pairs."""
    pairs = []
    for ch in text:
        if ch in '，。？！；：、':
            if pairs:
                pairs[-1] = (pairs[-1][0], LABEL2ID[ch])
        elif ch not in '\r\n""''()[]《》—…%·':
            pairs.append((ch, 0))
    return pairs

def load_month(month_num):
    """Load one month's articles."""
    fname = f"1998-{month_num:02d}-2003版-带音.txt"
    with open(CORPUS_DIR / fname, 'r', encoding='gb18030', errors='replace') as f:
        lines = f.readlines()

    articles = defaultdict(list)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        aid = parse_article_id(line)
        if aid:
            articles[aid].append(clean_line(line))

    # Extract pairs per article
    article_pairs = {}
    for aid in sorted(articles.keys()):
        text = ''.join(articles[aid])
        pairs = extract_chars_labels(text)
        if len(pairs) >= 32:  # Filter very short
            article_pairs[aid] = pairs

    print(f"Month {month_num}: {len(article_pairs)} articles")
    return article_pairs

def encode_article(pairs, codec):
    """Encode one article."""
    chars = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    char_ids = np.array([codec.encode_char(ch) for ch in chars], dtype=np.int32)
    label_ids = np.array(labels, dtype=np.int32)
    return char_ids, label_ids

def main():
    print("=== Preparing complete 1-6 month data ===")
    print("Split: 1-4月训练，5月验证，6月测试")

    # Load codec
    from punct.char_codec import CharCodec
    codec = CharCodec.from_bert_vocab('bert-base-chinese')

    # Load all months
    all_articles = {}
    for m in range(1, 7):
        all_articles[m] = load_month(m)

    # Split by month
    train_articles = {}
    for m in [1, 2, 3, 4]:
        train_articles.update(all_articles[m])

    dev_articles = all_articles[5]
    test_articles = all_articles[6]

    print(f"\nTotal: train={len(train_articles)}, dev={len(dev_articles)}, test={len(test_articles)}")

    # Encode
    def encode_split(articles_dict, name):
        chars_list = []
        labels_list = []
        total_chars = 0
        total_puncts = 0

        for aid in sorted(articles_dict.keys()):
            char_ids, label_ids = encode_article(articles_dict[aid], codec)
            chars_list.append(char_ids)
            labels_list.append(label_ids)
            total_chars += len(char_ids)
            total_puncts += int(np.sum(label_ids > 0))

        print(f"{name}: {len(chars_list)} articles, {total_chars} chars, {total_puncts} punctuation")
        return chars_list, labels_list

    print("\nEncoding...")
    train_char, train_label = encode_split(train_articles, "train")
    dev_char, dev_label = encode_split(dev_articles, "dev")
    test_char, test_label = encode_split(test_articles, "test")

    # Save
    np.save(OUT_DIR / "train_articles_char.npy", np.array(train_char, dtype=object), allow_pickle=True)
    np.save(OUT_DIR / "train_articles_label.npy", np.array(train_label, dtype=object), allow_pickle=True)
    np.save(OUT_DIR / "dev_articles_char.npy", np.array(dev_char, dtype=object), allow_pickle=True)
    np.save(OUT_DIR / "dev_articles_label.npy", np.array(dev_label, dtype=object), allow_pickle=True)
    np.save(OUT_DIR / "test_articles_char.npy", np.array(test_char, dtype=object), allow_pickle=True)
    np.save(OUT_DIR / "test_articles_label.npy", np.array(test_label, dtype=object), allow_pickle=True)

    codec.save(OUT_DIR / "char_codec.json")
    print(f"\nSaved to {OUT_DIR}/")

    # Save dataset statistics
    print("\n保存数据集统计...")
    def compute_label_dist(labels_list):
        all_labels = np.concatenate(labels_list)
        counts = np.bincount(all_labels, minlength=8)
        return {int(i): int(c) for i, c in enumerate(counts)}

    def get_split_stats(chars_list, labels_list, name):
        total_chars = sum(len(c) for c in chars_list)
        total_puncts = sum(int(np.sum(l > 0)) for l in labels_list)
        return {
            "n_articles": len(chars_list),
            "total_chars": total_chars,
            "total_punctuation": total_puncts,
            "label_distribution": compute_label_dist(labels_list),
            "avg_article_length": total_chars / len(chars_list) if chars_list else 0,
        }

    stats = {
        "dataset": "人民日报1998年1-6月",
        "splits": {
            "train": get_split_stats(train_char, train_label, "train"),
            "dev": get_split_stats(dev_char, dev_label, "dev"),
            "test": get_split_stats(test_char, test_label, "test"),
        },
        "vocab_size": len(codec.char2id),
        "labels": LABELS,
    }

    stats_file = OUT_DIR / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计已保存: {stats_file}")

    # Compute SHA-256 hashes for data integrity
    print("\nComputing SHA-256 hashes...")
    import hashlib
    import json

    hashes = {}
    files_to_hash = [
        "char_codec.json",
        "train_articles_char.npy",
        "train_articles_label.npy",
        "dev_articles_char.npy",
        "dev_articles_label.npy",
        "test_articles_char.npy",
        "test_articles_label.npy",
    ]

    for filename in files_to_hash:
        filepath = OUT_DIR / filename
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        hashes[filename] = sha256_hash.hexdigest()
        print(f"  {filename}: {sha256_hash.hexdigest()[:16]}...")

    hash_file = OUT_DIR / "data_hashes.json"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)
    print(f"\nHashes saved to {hash_file}")

if __name__ == '__main__':
    main()
