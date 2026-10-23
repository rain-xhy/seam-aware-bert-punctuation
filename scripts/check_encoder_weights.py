"""Verify the encoder weights changed during fine-tuning.

This reads a checkpoint saved by train.py (a state_dict) and compares one
layer of the BERT encoder's parameters against the pretrained base model. If
they are identical within tolerance, fine-tuning may have failed to update
the encoder, which would be a training bug.

    python -m scripts.check_encoder_weights checkpoints/T1_varlen/best.pt
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import BertForTokenClassification

from punct.model import encoder_weights_changed, get_weight_check_tensor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to best.pt (state_dict)")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    base = BertForTokenClassification.from_pretrained(args.bert_dir, num_labels=8, local_files_only=True)
    before = get_weight_check_tensor(base)

    finetuned = BertForTokenClassification.from_pretrained(args.bert_dir, num_labels=8, local_files_only=True)
    finetuned.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    after = get_weight_check_tensor(finetuned)

    changed, max_diff = encoder_weights_changed(before, after, args.atol)
    print(f"encoder weight check (bert.encoder.layer.0.attention.self.query.weight):")
    print(f"  max abs diff: {max_diff:.6e}")
    print(f"  changed: {changed}")
    if not changed:
        print("WARNING: encoder weights did not change (only the classifier head was updated)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
