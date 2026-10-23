"""Parallel batch inference with sliding windows (Task 2).

完善方案.md §4: split long text into overlapping windows (400-char input,
300-char attribution zone), run all windows in a single batched forward pass,
and merge predictions by assigning each char to exactly one window's output.

Usage:
  python infer.py --checkpoint checkpoints/T1/best.pt --input_file test.txt --output_file test_pred.txt
"""
import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
from transformers import BertTokenizer

from punct.char_codec import CharCodec


def sliding_windows(text: str, window_len: int = 400, stride: int = 300) -> List[Tuple[int, int]]:
    """Generate (start, end) pairs for sliding windows. Each char belongs to
    exactly one attribution zone (the window whose attribution zone contains it)."""
    n = len(text)
    if n == 0:
        return []
    if n <= window_len:
        return [(0, n)]

    windows = []
    pos = 0
    while pos < n:
        end = min(pos + window_len, n)
        windows.append((pos, end))
        pos += stride
        if end == n:
            break
    return windows


def batch_infer(
    model: torch.nn.Module,
    codec: CharCodec,
    tokenizer: BertTokenizer,
    texts: List[str],
    device: str = 'cuda',
    batch_size: int = 16
) -> List[str]:
    """Run batched inference on a list of text windows. Returns predicted
    punctuated strings (one per input text)."""
    model.eval()
    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Encode
        char_ids_batch = [codec.encode(t) for t in batch_texts]
        max_len = max(len(cids) for cids in char_ids_batch)

        input_ids = torch.full((len(batch_texts), max_len + 2), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_texts), max_len + 2), dtype=torch.long)

        for row, cids in enumerate(char_ids_batch):
            input_ids[row, 0] = tokenizer.cls_token_id
            input_ids[row, 1:1 + len(cids)] = torch.tensor(cids, dtype=torch.long)
            input_ids[row, 1 + len(cids)] = tokenizer.sep_token_id
            attention_mask[row, :2 + len(cids)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, 8)

        # Decode
        for row, (text, cids) in enumerate(zip(batch_texts, char_ids_batch)):
            pred_labels = logits[row, 1:1 + len(cids)].argmax(dim=-1).cpu().numpy()
            pred_text = codec.decode(cids, pred_labels.tolist())
            results.append(pred_text)

    return results


def infer_file(
    checkpoint_path: str,
    input_file: str,
    output_file: str,
    window_len: int = 400,
    stride: int = 300,
    batch_size: int = 16,
    device: str = 'cuda'
):
    """Infer on a whole file with sliding windows and batched forward passes."""
    # Load model
    from transformers import BertForTokenClassification
    model = BertForTokenClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
    codec = CharCodec(tokenizer.vocab)

    # Load input
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Generate windows
    windows = sliding_windows(full_text, window_len, stride)
    window_texts = [full_text[s:e] for s, e in windows]

    # Batch infer
    t0 = time.perf_counter()
    pred_windows = batch_infer(model, codec, tokenizer, window_texts, device, batch_size)
    t1 = time.perf_counter()

    # Merge: each char belongs to one window (the one whose attribution zone it's in)
    # For simplicity, attribution zone = [start+50, start+350) for all windows except first/last
    # First window: [0, 300), last window: [start, n)
    final_text = []
    n = len(full_text)
    char_coverage = [False] * n

    for idx, ((start, end), pred_window) in enumerate(zip(windows, pred_windows)):
        if len(windows) == 1:
            # Single window covers all
            attr_start, attr_end = 0, n
        elif idx == 0:
            # First window: attribute [0, stride)
            attr_start, attr_end = 0, min(stride, end)
        elif idx == len(windows) - 1:
            # Last window: attribute [start, n)
            attr_start, attr_end = start, n
        else:
            # Middle window: attribute [start, start+stride)
            attr_start, attr_end = start, min(start + stride, end)

        # Extract attributed portion from predicted window
        # pred_window corresponds to full_text[start:end]
        # We want pred_window[attr_start - start : attr_end - start]
        pred_attr = pred_window[attr_start - start : attr_end - start]
        final_text.append(pred_attr)

        for pos in range(attr_start, attr_end):
            char_coverage[pos] = True

    # Check all chars covered
    if not all(char_coverage):
        uncovered = [i for i, c in enumerate(char_coverage) if not c]
        print(f"Warning: {len(uncovered)} chars not covered by any window: {uncovered[:10]}")

    result = ''.join(final_text)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)

    # Stats
    n_chars = len(full_text)
    n_windows = len(windows)
    elapsed = t1 - t0
    throughput = n_chars / elapsed if elapsed > 0 else 0

    print(f"Inference complete:")
    print(f"  Input: {n_chars} chars, {n_windows} windows")
    print(f"  Time: {elapsed:.3f}s, {throughput:.1f} chars/s")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint dir or .pt file')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--window_len', type=int, default=400)
    parser.add_argument('--stride', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    infer_file(
        args.checkpoint,
        args.input_file,
        args.output_file,
        args.window_len,
        args.stride,
        args.batch_size,
        args.device
    )
