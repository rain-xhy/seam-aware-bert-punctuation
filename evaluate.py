"""Simple evaluation script for demo."""
import sys
from punct.metrics import punctuation_prf, sentence_final_f1
import numpy as np

LABELS = ['O', '，', '。', '？', '！', '；', '：', '、']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def extract_labels(text_with_punct):
    """Extract (char, label) pairs from punctuated text."""
    pairs = []
    for ch in text_with_punct:
        if ch in LABEL2ID and ch != 'O':
            # Punctuation: attach to previous char
            if pairs:
                pairs[-1] = (pairs[-1][0], LABEL2ID[ch])
        else:
            # Regular char
            pairs.append((ch, 0))
    return pairs

def compare_files(gold_file, pred_file):
    """Compare gold and predicted punctuation."""
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_text = f.read()
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_text = f.read()

    gold_pairs = extract_labels(gold_text)
    pred_pairs = extract_labels(pred_text)

    # Ensure same length
    min_len = min(len(gold_pairs), len(pred_pairs))
    gold_labels = np.array([p[1] for p in gold_pairs[:min_len]], dtype=np.int32)
    pred_labels = np.array([p[1] for p in pred_pairs[:min_len]], dtype=np.int32)

    # Compute metrics
    prf = punctuation_prf(gold_labels, pred_labels, num_labels=8, none_label=0)
    sf = sentence_final_f1(gold_labels, pred_labels, sentence_final_labels=[2, 3, 4])

    print("=== Evaluation Results ===")
    print(f"Positions: {len(gold_labels)}")
    print(f"Accuracy: {prf['accuracy']:.4f}")
    print(f"Micro-F1: {prf['micro_f1']:.4f}")
    print(f"Macro-F1: {prf['macro_f1']:.4f}")
    print(f"Sentence-final F1: {sf['f1']:.4f}")
    print("\nPer-class F1:")
    for label_id, stats in prf['per_class'].items():
        print(f"  {LABELS[label_id]:>2s}: F1={stats['f1']:.4f}, support={stats['support']}")

    return prf

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <gold_file> <pred_file>")
        sys.exit(1)

    compare_files(sys.argv[1], sys.argv[2])
