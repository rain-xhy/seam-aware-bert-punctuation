#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process full 1998 Jan-Jun corpus, split into train/dev/test
"""
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter

CORPUS_DIR = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\1998-上半年-带音")
OUT_DIR = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\prepared_data_full")
OUT_DIR.mkdir(exist_ok=True)

# Target punctuation (Chinese full-width)
LABELS = ['O', '，', '。', '？', '！', '；', '：', '、']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def clean_text(text):
    """Remove pinyin annotations {shang5}, keep target punctuation"""
    # Remove pinyin
    text = re.sub(r'\{[^}]+\}', '', text)
    # Remove article numbers like 19980101-01-001-001
    text = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}/[mne]$', '', text, flags=re.MULTILINE)
    return text

def extract_chars_labels(text):
    """Extract (char, label) pairs"""
    pairs = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        i = 0
        while i < len(line):
            ch = line[i]
            if ch in '，。？！；：、,.?!;:""''()[]《》—…%·':
                # Punctuation
                if ch in '，。？！；：、':
                    # Target punctuation: attach to previous char
                    if pairs:
                        pairs[-1] = (pairs[-1][0], ch)
                # Skip non-target punctuation
                i += 1
            else:
                # Content char
                pairs.append((ch, 'O'))
                i += 1
    return pairs

def load_month(month_num):
    """Load one month's corpus"""
    fname = f"1998-{month_num:02d}-2003版-带音.txt"
    fpath = CORPUS_DIR / fname
    with open(fpath, 'r', encoding='gb18030', errors='replace') as f:
        text = f.read()
    text = clean_text(text)
    pairs = extract_chars_labels(text)
    print(f"Month {month_num}: {len(pairs)} char-label pairs")
    return pairs

def main():
    all_pairs = []
    for m in range(1, 7):
        pairs = load_month(m)
        all_pairs.extend(pairs)

    print(f"Total: {len(all_pairs)} pairs")

    # Label distribution
    label_cnt = Counter(l for c, l in all_pairs)
    print("Label distribution:")
    for lbl in LABELS:
        print(f"  {lbl:>3s}: {label_cnt[lbl]:>8d} ({100*label_cnt[lbl]/len(all_pairs):.2f}%)")

    # Split: 90% train, 5% dev, 5% test
    np.random.seed(42)
    N = len(all_pairs)
    indices = np.random.permutation(N)
    n_train = int(0.9 * N)
    n_dev = int(0.05 * N)

    train_idx = indices[:n_train]
    dev_idx = indices[n_train:n_train+n_dev]
    test_idx = indices[n_train+n_dev:]

    def save_split(name, idx):
        chars = np.array([all_pairs[i][0] for i in idx], dtype='U1')
        labels = np.array([LABEL2ID[all_pairs[i][1]] for i in idx], dtype=np.int8)
        np.save(OUT_DIR / f"{name}_char.npy", chars)
        np.save(OUT_DIR / f"{name}_label.npy", labels)
        print(f"{name}: {len(idx)} pairs saved")

    save_split('train', train_idx)
    save_split('dev', dev_idx)
    save_split('test', test_idx)

    # Save codec
    codec = {
        'labels': LABELS,
        'label2id': LABEL2ID,
        'stats': {lbl: int(label_cnt[lbl]) for lbl in LABELS}
    }
    with open(OUT_DIR / 'char_codec.json', 'w', encoding='utf-8') as f:
        json.dump(codec, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {OUT_DIR}")

if __name__ == '__main__':
    main()
