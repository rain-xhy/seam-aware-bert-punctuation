#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create minimal dataset for testing (100 articles, 1 epoch)."""
import numpy as np
from pathlib import Path

# Load original data
data_dir = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\prepared_data")
out_dir = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\prepared_data_mini")
out_dir.mkdir(exist_ok=True)

print("Loading original data...")
train_char = np.load(data_dir / "train_articles_char.npy", allow_pickle=True)
train_label = np.load(data_dir / "train_articles_label.npy", allow_pickle=True)
dev_char = np.load(data_dir / "dev_articles_char.npy", allow_pickle=True)
dev_label = np.load(data_dir / "dev_articles_label.npy", allow_pickle=True)

print(f"Original: train={len(train_char)}, dev={len(dev_char)}")

# Take first 100 articles for train, 20 for dev
mini_train_char = train_char[:100]
mini_train_label = train_label[:100]
mini_dev_char = dev_char[:20]
mini_dev_label = dev_label[:20]

print(f"Mini: train={len(mini_train_char)}, dev={len(mini_dev_char)}")

# Save
np.save(out_dir / "train_articles_char.npy", mini_train_char)
np.save(out_dir / "train_articles_label.npy", mini_train_label)
np.save(out_dir / "dev_articles_char.npy", mini_dev_char)
np.save(out_dir / "dev_articles_label.npy", mini_dev_label)

# Copy codec
import shutil
shutil.copy(data_dir / "char_codec.json", out_dir / "char_codec.json")

print(f"Saved to {out_dir}")
print("Train stats:")
total_chars = sum(len(a) for a in mini_train_char)
total_puncts = sum(np.sum(a > 0) for a in mini_train_label)
print(f"  {total_chars} chars, {total_puncts} punctuation marks")
