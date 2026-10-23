"""Quick training test: estimate full training time.

Use 100 articles (~1% of training set) for 1 epoch to estimate full training time.
Training set: 12128 articles, 7125456 chars.
This test: 100 articles, ~58k chars.

Usage:
    conda activate yolov8
    python -X utf8 quick_train_test.py ../1998-上半年-带音
"""
import sys, time, glob, os, re
from collections import defaultdict

# Minimal data loading (same as split_probe.py)
ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
MONTH_RE = re.compile(r"1998-(\d\d)")
SPLIT = {"01": "train", "02": "train", "03": "train", "04": "train"}

def load_sample(corpus_dir, n_articles=100):
    """Load first n_articles from training months."""
    articles = []
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        month = MONTH_RE.search(os.path.basename(f))
        if not month or SPLIT.get(month.group(1)) != "train":
            continue
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                if len(articles) >= n_articles:
                    break
                line = line.rstrip("\r\n")
                if not line.strip() or not ID_RE.match(line):
                    continue
                # Quick clean: remove pinyin, spaces
                body = PINYIN_RE.sub("", ID_RE.sub("", line, count=1)).replace(" ", "").replace("　", "")
                if len(body) > 10:
                    articles.append(body)
        if len(articles) >= n_articles:
            break
    return articles

def main(corpus_dir):
    print(f"Loading {100} sample articles...")
    articles = load_sample(corpus_dir, 100)
    n_chars = sum(len(a) for a in articles)
    print(f"Loaded {len(articles)} articles, {n_chars} chars")
    print()

    # Simulate training loop (no actual training, just data processing)
    print("Simulating training loop (data processing only)...")
    start = time.time()

    # Simulate: tokenize + random crop
    for i, text in enumerate(articles):
        # Simulate random crop (32-510 chars)
        import random
        max_len = min(len(text), 510)
        if max_len > 32:
            crop_len = random.randint(32, max_len)
            start_pos = random.randint(0, len(text) - crop_len)
            _ = text[start_pos:start_pos + crop_len]
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(articles)} articles")

    elapsed = time.time() - start
    print(f"\nData processing: {elapsed:.2f}s for {n_chars} chars")
    print(f"Speed: {n_chars / elapsed:.0f} chars/s")
    print()

    # Estimate full training time
    total_chars = 7125456
    total_articles = 12128
    epochs = 3

    # Assume actual training (GPU forward+backward) is 10x slower than data processing
    training_multiplier = 10

    estimated_per_epoch = (elapsed / n_chars) * total_chars * training_multiplier
    estimated_total = estimated_per_epoch * epochs

    print("=" * 60)
    print(f"Estimated training time (conservative, 10x multiplier):")
    print(f"  Per epoch: {estimated_per_epoch / 60:.1f} min")
    print(f"  Total (3 epochs): {estimated_total / 60:.1f} min ({estimated_total / 3600:.2f} hours)")
    print()

    if estimated_total > 7200:  # > 2 hours
        sample_ratio = 7200 / estimated_total
        print(f"WARNING: Estimated > 2 hours. Consider sampling to {sample_ratio:.1%} of training set.")
    elif estimated_total > 3600:  # > 1 hour
        print(f"INFO: Estimated > 1 hour. Acceptable, but sampling is an option.")
    else:
        print(f"INFO: Estimated < 1 hour. No sampling needed.")
    print("=" * 60)
    print()
    print("NOTE: This is a rough estimate. Actual training time depends on:")
    print("  - GPU utilization (batch size, gradient accumulation)")
    print("  - Data loading overhead")
    print("  - Checkpoint saving")
    print("  - Validation runs")
    print()
    print("Next: Run actual training on 1% sample to get real GPU timing.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_train_test.py <corpus_dir>")
        sys.exit(1)
    main(sys.argv[1])
