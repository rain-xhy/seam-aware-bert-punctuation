"""Complete inference script with all methods.

Usage:
    python run_inference.py --method sequential --checkpoint checkpoints/T1/best.pt --input test.txt --output pred.txt
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import BertForTokenClassification

from punct.char_codec import CharCodec
from punct.inference import (
    Window,
    InferenceConfig,
    create_windows_direct_cut,
    create_windows_sequential,
    create_windows_tang_sequential,
    create_windows_parallel,
    create_windows_tang_A,
    merge_overlapping_predictions,
    merge_tang_A,
    identify_low_confidence_seams,
    create_patch_windows,
    compute_TER,
    compute_coverage,
    softmax,
)


class InferenceEngine:
    """Unified inference engine for all methods."""

    def __init__(self, checkpoint_path: str, codec_path: str, device: str = 'cuda'):
        self.device = torch.device(device)

        # Check if checkpoint_path is a .pt file or a directory
        if checkpoint_path.endswith('.pt'):
            # Load from state_dict
            import os
            bert_dir = 'bert-base-chinese'  # Use base model
            self.model = BertForTokenClassification.from_pretrained(bert_dir, num_labels=8)
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            # Load from directory (Hugging Face format)
            self.model = BertForTokenClassification.from_pretrained(checkpoint_path, num_labels=8)

        self.model.to(self.device)
        self.model.eval()

        self.codec = CharCodec.load(codec_path)
        self.LABELS = ['O', '，', '。', '？', '！', '；', '：', '、']

    def encode_windows(self, windows: List[Window]) -> Dict:
        """Encode windows to model inputs."""
        batch_size = len(windows)
        max_len = max(len(w.text) for w in windows) + 2  # +2 for [CLS], [SEP]

        input_ids = np.full((batch_size, max_len), self.codec.pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, win in enumerate(windows):
            char_ids = self.codec.encode(win.text)
            n = len(char_ids)
            input_ids[i, 0] = self.codec.cls_id
            input_ids[i, 1:1 + n] = char_ids
            input_ids[i, 1 + n] = self.codec.sep_id
            attention_mask[i, :2 + n] = 1

        return {
            'input_ids': torch.from_numpy(input_ids).to(self.device),
            'attention_mask': torch.from_numpy(attention_mask).to(self.device)
        }

    def forward_windows(self, windows: List[Window], batch_size: int = 32) -> List[np.ndarray]:
        """Run forward pass on windows in batches.

        Returns:
            logits_list: [num_windows, window_len, num_labels]
        """
        all_logits = []

        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            inputs = self.encode_windows(batch_windows)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()  # [B, max_len, 8]

            # Extract valid logits (skip [CLS] and padding)
            for j, win in enumerate(batch_windows):
                win_len = len(win.text)
                all_logits.append(logits[j, 1:1 + win_len, :])  # [win_len, 8]

        return all_logits

    def decode_labels(self, text: str, labels: np.ndarray) -> str:
        """Convert label IDs back to text with punctuation."""
        result = []
        for char, label_id in zip(text, labels):
            result.append(char)
            if label_id > 0:
                result.append(self.LABELS[label_id])
        return ''.join(result)

    # === Inference methods ===

    def infer_direct_cut(self, text: str, window_size: int = 64) -> Dict:
        """Method 0: Direct cut, no overlap."""
        start_time = time.time()

        windows = create_windows_direct_cut(text, window_size)
        logits_list = self.forward_windows(windows)

        # Simple concatenation (no overlap)
        all_labels = []
        for logits in logits_list:
            all_labels.extend(np.argmax(logits, axis=-1).tolist())

        elapsed = time.time() - start_time
        result_text = self.decode_labels(text, np.array(all_labels[:len(text)]))

        return {
            'method': 'direct_cut',
            'text': result_text,
            'num_windows': len(windows),
            'TER': compute_TER(windows, len(text)),
            'coverage': compute_coverage(windows, len(text)),
            'time': elapsed,
            'throughput': len(text) / elapsed
        }

    def infer_sequential(self, text: str, window_size: int = 400) -> Dict:
        """Method 1: Sequential sliding (俞敬松2019)."""
        start_time = time.time()

        # Sequential: process one window at a time
        windows = create_windows_sequential(text, window_size)
        all_labels = np.zeros(len(text), dtype=np.int32)

        for win in windows:
            inputs = self.encode_windows([win])
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, 1:1 + len(win.text), :].cpu().numpy()
            pred_labels = np.argmax(logits, axis=-1)
            all_labels[win.start:win.end] = pred_labels

        elapsed = time.time() - start_time
        result_text = self.decode_labels(text, all_labels)

        return {
            'method': 'sequential',
            'text': result_text,
            'num_windows': len(windows),
            'TER': compute_TER(windows, len(text)),
            'time': elapsed,
            'throughput': len(text) / elapsed
        }

    def infer_parallel_basic(self, text: str, cfg: InferenceConfig) -> Dict:
        """Method B baseline: Parallel + hard masking + argmax."""
        start_time = time.time()

        windows = create_windows_parallel(text, cfg.window_size, cfg.stride, cfg.short_tail)
        logits_list = self.forward_windows(windows)

        merged_labels = merge_overlapping_predictions(
            text, windows, logits_list,
            mask_left=cfg.mask_left,
            mask_right=cfg.mask_right,
            merge_strategy='argmax'
        )

        elapsed = time.time() - start_time
        result_text = self.decode_labels(text, merged_labels)

        return {
            'method': 'parallel_basic',
            'text': result_text,
            'num_windows': len(windows),
            'TER': compute_TER(windows, len(text)),
            'coverage': compute_coverage(windows, len(text)),
            'time': elapsed,
            'throughput': len(text) / elapsed,
            'config': {
                'W': cfg.window_size,
                'stride': cfg.stride,
                'mask_left': cfg.mask_left,
                'mask_right': cfg.mask_right,
                'short_tail': cfg.short_tail
            }
        }

    def infer_method_B(self, text: str, cfg: InferenceConfig, merge_strategy: str = 'weighted') -> Dict:
        """Method B: MF-PW with weighted merging."""
        start_time = time.time()

        windows = create_windows_parallel(text, cfg.window_size, cfg.stride, cfg.short_tail)
        logits_list = self.forward_windows(windows)

        merged_labels = merge_overlapping_predictions(
            text, windows, logits_list,
            mask_left=cfg.mask_left,
            mask_right=cfg.mask_right,
            merge_strategy=merge_strategy
        )

        elapsed = time.time() - start_time
        result_text = self.decode_labels(text, merged_labels)

        return {
            'method': 'method_B',
            'text': result_text,
            'num_windows': len(windows),
            'TER': compute_TER(windows, len(text)),
            'time': elapsed,
            'throughput': len(text) / elapsed,
            'config': vars(cfg)
        }

    def infer_method_C(
        self,
        text: str,
        cfg: InferenceConfig,
        confidence_threshold: float = 0.1,
        seam_margin: int = 8,
        max_patches: int = 50
    ) -> Dict:
        """Method C: SA-PW with confidence-guided patching."""
        start_time = time.time()

        # Pass 1: Coarse windows with small overlap
        cfg_pass1 = InferenceConfig(
            window_size=cfg.window_size,
            stride=cfg.window_size - 20,  # 更小的重叠
            mask_left=cfg.mask_left,
            mask_right=cfg.mask_right,
            short_tail=cfg.short_tail
        )

        windows_pass1 = create_windows_parallel(text, cfg_pass1.window_size, cfg_pass1.stride, cfg_pass1.short_tail)
        logits_pass1 = self.forward_windows(windows_pass1)

        # Merge pass 1
        merged_labels_pass1 = merge_overlapping_predictions(
            text, windows_pass1, logits_pass1,
            mask_left=cfg_pass1.mask_left,
            mask_right=cfg_pass1.mask_right,
            merge_strategy='weighted'
        )

        # Compute probabilities for confidence estimation
        merged_probs_pass1 = np.zeros((len(text), 8), dtype=np.float32)
        for i, logits in enumerate(logits_pass1):
            win = windows_pass1[i]
            probs = softmax(logits, axis=-1)
            for j in range(len(win.text)):
                pos = win.start + j
                if 0 <= pos < len(text):
                    merged_probs_pass1[pos] += probs[j]
        # Normalize
        for i in range(len(text)):
            s = np.sum(merged_probs_pass1[i])
            if s > 0:
                merged_probs_pass1[i] /= s

        # Identify low-confidence seams
        seam_positions = identify_low_confidence_seams(
            merged_probs_pass1,
            windows_pass1,
            cfg_pass1.mask_left,
            cfg_pass1.mask_right,
            threshold=confidence_threshold,
            seam_margin=seam_margin
        )

        # Limit number of patches
        if len(seam_positions) > max_patches:
            # Sort by confidence (lower = higher priority)
            confidences = [np.max(merged_probs_pass1[pos]) - np.sort(merged_probs_pass1[pos])[-2]
                          for pos in seam_positions]
            sorted_indices = np.argsort(confidences)[:max_patches]
            seam_positions = [seam_positions[i] for i in sorted_indices]

        # Pass 2: Create patch windows
        if len(seam_positions) > 0:
            patch_windows = create_patch_windows(text, seam_positions, cfg.window_size)
            logits_patches = self.forward_windows(patch_windows)

            # Merge patches into pass1 result (overwrite low-confidence positions)
            for win, logits in zip(patch_windows, logits_patches):
                pred_labels = np.argmax(logits, axis=-1)
                for i in range(len(win.text)):
                    pos = win.start + i
                    if pos in seam_positions:
                        merged_labels_pass1[pos] = pred_labels[i]

        elapsed = time.time() - start_time
        result_text = self.decode_labels(text, merged_labels_pass1)

        total_windows = windows_pass1 + (patch_windows if len(seam_positions) > 0 else [])

        return {
            'method': 'method_C',
            'text': result_text,
            'num_windows_pass1': len(windows_pass1),
            'num_patches': len(seam_positions),
            'TER': compute_TER(total_windows, len(text)),
            'time': elapsed,
            'throughput': len(text) / elapsed,
            'config': {
                'confidence_threshold': confidence_threshold,
                'seam_margin': seam_margin,
                'max_patches': max_patches
            }
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=[
        'direct_cut', 'sequential', 'parallel_basic', 'method_B', 'method_C'
    ])
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--codec', default='prepared_data/char_codec.json')
    parser.add_argument('--input', required=True, help='Input text file')
    parser.add_argument('--output', required=True, help='Output text file')
    parser.add_argument('--window_size', type=int, default=400)
    parser.add_argument('--stride', type=int, default=300)
    parser.add_argument('--mask_left', type=int, default=50)
    parser.add_argument('--mask_right', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    # Initialize engine
    engine = InferenceEngine(args.checkpoint, args.codec, args.device)

    # Run inference
    cfg = InferenceConfig(
        window_size=args.window_size,
        stride=args.stride,
        mask_left=args.mask_left,
        mask_right=args.mask_right,
        short_tail=True
    )

    if args.method == 'direct_cut':
        result = engine.infer_direct_cut(text, window_size=64)
    elif args.method == 'sequential':
        result = engine.infer_sequential(text, window_size=args.window_size)
    elif args.method == 'parallel_basic':
        result = engine.infer_parallel_basic(text, cfg)
    elif args.method == 'method_B':
        result = engine.infer_method_B(text, cfg, merge_strategy='weighted')
    elif args.method == 'method_C':
        result = engine.infer_method_C(text, cfg)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save output
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    # Save stats
    stats_path = args.output.replace('.txt', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"Method: {result['method']}")
    print(f"Windows: {result.get('num_windows', 'N/A')}")
    print(f"TER: {result.get('TER', 0):.3f}")
    print(f"Time: {result['time']:.2f}s")
    print(f"Throughput: {result['throughput']:.1f} chars/s")
    print(f"Output: {args.output}")
    print(f"Stats: {stats_path}")


if __name__ == '__main__':
    main()
