"""Real training test: 1% data, 1 epoch, measure actual GPU time.

Use ~120 articles (~1% of 12128) to run actual BERT training for 1 epoch.
Measure real GPU forward+backward time, estimate full training.

Usage:
    conda activate yolov8
    python -X utf8 real_train_test.py ../1998-上半年-带音
"""
import sys, time, glob, os, re, random
import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW

# Data loading (same as split_probe.py)
ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
MONTH_RE = re.compile(r"1998-(\d\d)")
SPLIT = {"01": "train", "02": "train", "03": "train", "04": "train"}

PUNCT_MAP = {"，": 1, "。": 2, "？": 3, "！": 4, "；": 5, "：": 6, "、": 7}

def load_sample(corpus_dir, n_articles=120):
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
                body = PINYIN_RE.sub("", ID_RE.sub("", line, count=1)).replace(" ", "").replace("　", "")
                if len(body) > 10:
                    articles.append(body)
        if len(articles) >= n_articles:
            break
    return articles

def extract_labels(text):
    """Extract input (no punct) and labels (0-7)."""
    input_chars = []
    labels = []
    for ch in text:
        if ch in PUNCT_MAP:
            if input_chars:
                labels[-1] = PUNCT_MAP[ch]
        else:
            input_chars.append(ch)
            labels.append(0)
    return "".join(input_chars), labels

def main(corpus_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be very slow!")
        return

    print(f"\nLoading ~120 sample articles (1% of training set)...")
    articles = load_sample(corpus_dir, 120)
    n_chars = sum(len(a) for a in articles)
    print(f"Loaded {len(articles)} articles, {n_chars:,} chars")

    # Prepare samples
    print("\nPreparing samples (extract labels)...")
    samples = []
    for text in articles:
        input_text, labels = extract_labels(text)
        if len(input_text) > 10:
            samples.append((input_text, labels))
    print(f"Prepared {len(samples)} samples")

    # Load model
    print("\nLoading bert-base-chinese...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=8)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Training loop
    print("\nRunning 1 epoch (batch 8, gradient accumulation 2)...")
    print("This will take a few minutes...")

    start_time = time.time()
    total_loss = 0
    n_steps = 0

    batch_size = 8
    grad_accum = 2
    effective_batch = batch_size * grad_accum

    batch_inputs = []
    batch_labels = []

    for i, (input_text, labels) in enumerate(samples):
        # Random crop (32-256 for speed)
        max_len = min(len(input_text), 256)
        if max_len > 32:
            crop_len = random.randint(32, max_len)
            start_pos = random.randint(0, len(input_text) - crop_len)
            input_text = input_text[start_pos:start_pos + crop_len]
            labels = labels[start_pos:start_pos + crop_len]

        batch_inputs.append(input_text)
        batch_labels.append(labels)

        if len(batch_inputs) == batch_size:
            # Tokenize
            encoding = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Align labels
            max_seq_len = input_ids.size(1)
            aligned_labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
            for j, lbl in enumerate(batch_labels):
                for k, l in enumerate(lbl[:max_seq_len - 2]):  # -2 for [CLS] [SEP]
                    aligned_labels[j, k + 1] = l
            aligned_labels = aligned_labels.to(device)

            # Forward + backward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=aligned_labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            total_loss += loss.item() * grad_accum

            if (n_steps + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            n_steps += 1
            batch_inputs = []
            batch_labels = []

            if n_steps % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {n_steps}/{len(samples)//batch_size}, loss={total_loss/n_steps:.4f}, elapsed={elapsed:.1f}s")

    elapsed_total = time.time() - start_time

    print(f"\nCompleted {n_steps} steps in {elapsed_total:.1f}s")
    print(f"Average loss: {total_loss / n_steps:.4f}")
    print(f"Speed: {n_chars / elapsed_total:.0f} chars/s")

    # Estimate full training
    total_chars = 7_125_456
    epochs = 3

    estimated_per_epoch = (elapsed_total / n_chars) * total_chars
    estimated_total = estimated_per_epoch * epochs

    print("\n" + "=" * 60)
    print(f"Estimated full training time:")
    print(f"  Per epoch: {estimated_per_epoch / 60:.1f} min")
    print(f"  Total (3 epochs): {estimated_total / 60:.1f} min ({estimated_total / 3600:.2f} hours)")
    print()

    if estimated_total > 7200:  # > 2 hours
        sample_ratio = 7200 / estimated_total
        print(f"⚠️  > 2 hours. Recommend sampling to {sample_ratio:.0%} of training set.")
    elif estimated_total > 3600:  # > 1 hour
        print(f"ℹ️  > 1 hour. Acceptable, or sample to ~50% for faster iteration.")
    else:
        print(f"✓ < 1 hour. No sampling needed.")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python real_train_test.py <corpus_dir>")
        sys.exit(1)
    main(sys.argv[1])
