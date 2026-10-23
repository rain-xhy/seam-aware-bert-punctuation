"""Train a BertForTokenClassification model on 1998-01 人民日报, following
完善方案.md §5.1. Usage:

  python train.py --data_dir <path_to_prepared_data> --out_dir <checkpoint_dir>

  --data_dir must contain train_meta.json, val_meta.json, all_windows.bin (or
  train_windows.bin/val_windows.bin), article_char_ids.bin, article_labels.bin.
  The script loads windows, runs 3 epochs, validates after each epoch, and saves
  the checkpoint with best val micro-F1 to <out_dir>/best.pt.

完善方案.md §5.1 specifies:
  - bert-base-chinese, full fine-tune (no frozen layers)
  - AdamW, lr=3e-5, weight_decay=0.01, grad_clip=1.0, warmup 10%, linear decay
  - batch=8, grad_accum=2 (effective batch 16), BF16 mixed precision
  - 3 epochs, validate after each epoch, save checkpoint with best val micro-F1
  - Compute val loss and metrics (P/R/F1 per class, micro/macro-F1, sentence-end F1)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification, get_linear_schedule_with_warmup

# ensure project root is in sys.path so we can import punct.*
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from punct.codec import CharCodec
from punct.data import WindowDataset, collate_windows
from punct.metrics import compute_metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, grad_accum_steps, grad_clip):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            # logits: [batch, seq, n_classes]; labels: [batch, seq]
            # ignore label=-100 (padding and special tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum_steps
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()  # [batch, seq]
            labels_np = labels.cpu().numpy()
            for p, l in zip(preds, labels_np):
                mask = l != -100
                all_preds.extend(p[mask].tolist())
                all_labels.extend(l[mask].tolist())
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bert_dir", default="./bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # load codec
    codec_path = os.path.join(args.data_dir, "char_codec.json")
    codec = CharCodec.load(codec_path)

    # load datasets
    train_dataset = WindowDataset(args.data_dir, split="train")
    val_dataset = WindowDataset(args.data_dir, split="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_windows, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows, num_workers=0
    )

    # load model
    model = BertForTokenClassification.from_pretrained(args.bert_dir, num_labels=8).to(device)

    # optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) // args.grad_accum * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Training {args.epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={args.grad_accum}")
    print(f"Total steps={total_steps}, warmup={warmup_steps}, device={device}")

    best_val_f1 = 0.0
    best_checkpoint = None
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args.grad_accum, args.grad_clip
        )
        val_loss, val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={elapsed:.1f}s")
        print(f"  Val micro-F1={val_metrics['micro_f1']:.4f} | macro-F1={val_metrics['macro_f1']:.4f}")
        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_checkpoint = os.path.join(args.out_dir, "best.pt")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  ✓ Saved best checkpoint to {best_checkpoint}")

    print(f"\nTraining finished. Best val micro-F1={best_val_f1:.4f}, checkpoint={best_checkpoint}")


if __name__ == "__main__":
    main()
