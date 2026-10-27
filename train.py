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

from punct.char_codec import CharCodec
from punct.dataset import WindowDataset, collate_windows
from punct.metrics import compute_metrics


def save_state_atomic(path, state):
    """Write to a temp file, then rename, so a crash while writing never
    destroys the previous good state."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def epoch_order(n_items, seed, epoch):
    """Shuffled sample order of one epoch, reproducible from (seed, epoch) so a
    resumed run can continue the same epoch from the middle."""
    g = torch.Generator()
    g.manual_seed(seed * 1000 + epoch)
    return torch.randperm(n_items, generator=g).tolist()


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, grad_accum_steps, grad_clip,
                    class_weight=None, log_every=2000, start_batch=0, n_total=None, total_loss=0.0,
                    save_every=0, on_save=None):
    """start_batch: micro-batches of this epoch already done (resume). The
    dataloader must yield only the remaining ones. on_save(done, total_loss) is
    called every `save_every` micro-batches, only right after an optimizer step."""
    model.train()
    window_loss = 0.0
    t_start = time.time()
    n_batches = n_total if n_total is not None else len(dataloader)
    optimizer.zero_grad()
    for j, batch in enumerate(dataloader):
        i = start_batch + j
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # logits: [batch, seq, n_classes]; labels: [batch, seq]
            # ignore label=-100 (padding and special tokens)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, weight=class_weight)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        stepped = (i + 1) % grad_accum_steps == 0
        if stepped:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        step_loss = loss.item() * grad_accum_steps
        total_loss += step_loss
        window_loss += step_loss
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (j + 1) * (n_batches - i - 1)
            print(f"  step {i+1}/{n_batches} | loss(last {log_every})={window_loss/log_every:.4f} "
                  f"| elapsed={elapsed/60:.1f}min | epoch_eta={eta/60:.1f}min", flush=True)
            window_loss = 0.0
        if save_every and on_save is not None and stepped and (i + 1) % save_every == 0 and i + 1 < n_batches:
            on_save(i + 1, total_loss)
    return total_loss / n_batches


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    print("  Validating...", flush=True)
    n_batches = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i > 0 and i % 500 == 0:
                print(f"    validation batch {i}/{n_batches}", flush=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()

            # Collect predictions and labels (flatten and filter)
            preds = logits.argmax(dim=-1).cpu().numpy()  # [batch, seq]
            labels_np = labels.cpu().numpy()
            for p, l in zip(preds, labels_np):
                mask = l != -100
                all_preds.extend(p[mask].tolist())
                all_labels.extend(l[mask].tolist())

    avg_loss = total_loss / len(dataloader)

    # Construct EvalPrediction-like object for compute_metrics
    # Reshape as 2D arrays (fake batch dimension)
    import numpy as np
    from collections import namedtuple
    EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])

    # Convert to arrays and reshape
    all_preds_arr = np.array(all_preds, dtype=np.int64).reshape(1, -1)  # [1, total_valid_tokens]
    all_labels_arr = np.array(all_labels, dtype=np.int64).reshape(1, -1)

    # Create fake logits (one-hot from preds)
    num_labels = 8
    fake_logits = np.zeros((1, len(all_preds), num_labels), dtype=np.float32)
    fake_logits[0, np.arange(len(all_preds)), all_preds_arr[0]] = 1.0

    eval_pred = EvalPrediction(
        predictions=fake_logits,
        label_ids=all_labels_arr
    )
    metrics = compute_metrics(eval_pred)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bert_dir", default="./bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=510, help="Maximum sequence length for training")
    parser.add_argument("--class_weights", action='store_true', help="Use class-weighted loss (frequency inverse sqrt)")
    parser.add_argument("--min_len", type=int, default=32,
                        help="Minimum crop length; set equal to --max_len for a fixed-length model")
    parser.add_argument("--edge_aware_prob", type=float, default=0.3,
                        help="Probability that a crop gets a shortened right edge (0 disables edge-aware training)")
    parser.add_argument("--center_only", action='store_true', help="Supervise only the centre of each crop")
    parser.add_argument("--center_margin", type=int, default=50,
                        help="Unsupervised chars on each side when --center_only is set")
    parser.add_argument("--save_every", type=int, default=4000,
                        help="Save a full resumable state every N micro-batches (0 disables)")
    parser.add_argument("--no_resume", action='store_true',
                        help="Ignore an existing resume.pt in --out_dir and start from scratch")
    args = parser.parse_args()
    if args.min_len > args.max_len:
        parser.error("--min_len must not exceed --max_len")
    sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # load codec
    codec_path = os.path.join(args.data_dir, "char_codec.json")
    codec = CharCodec.load(codec_path)

    # load datasets
    from punct.dataset import CropConfig
    train_cfg = CropConfig(min_len=args.min_len, max_len=args.max_len, edge_aware_prob=args.edge_aware_prob,
                           center_only_supervision=args.center_only, center_margin=args.center_margin)
    # validation always uses plain crops (no edge shortening, full supervision) so
    # that models trained with different switches are compared on the same signal
    val_cfg = CropConfig(min_len=args.min_len, max_len=args.max_len, edge_aware_prob=0.0)
    train_dataset = WindowDataset(args.data_dir, split="train", cfg=train_cfg, seed=args.seed)
    val_dataset = WindowDataset(args.data_dir, split="dev", cfg=val_cfg, seed=args.seed + 1)

    # collate function needs special token IDs
    def collate_fn(batch):
        return collate_windows(batch, codec.cls_id, codec.sep_id, codec.pad_id)

    n_train = len(train_dataset)
    batches_per_epoch = (n_train + args.batch_size - 1) // args.batch_size

    def make_train_loader(epoch, start_batch):
        """Loader over the not-yet-seen part of this epoch's reproducible order."""
        order = epoch_order(n_train, args.seed, epoch)[start_batch * args.batch_size:]
        train_dataset.rng = __import__("random").Random(args.seed * 7919 + epoch * 104729 + start_batch)
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, order), batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # load model
    model = BertForTokenClassification.from_pretrained(args.bert_dir, num_labels=8).to(device)

    # compute class weights if needed
    class_weight = None
    if args.class_weights:
        print("Computing class weights from training data...")
        label_counts = np.bincount(np.concatenate(list(train_dataset.articles_label)).astype(np.int64),
                                   minlength=8).astype(np.float64)[:8]
        # inverse square root of frequency relative to class 0 (no punctuation), capped at 8
        weights = np.sqrt(label_counts[0] / np.maximum(label_counts, 1.0))
        weights = np.minimum(weights, 8.0)
        weights[0] = 1.0
        class_weight = torch.from_numpy(weights).float().to(device)
        print(f"Class weights: {weights}")

    # optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = batches_per_epoch // args.grad_accum * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Training {args.epochs} epochs, {batches_per_epoch} batches/epoch, grad_accum={args.grad_accum}")
    print(f"Total steps={total_steps}, warmup={warmup_steps}, device={device}")

    best_val_f1 = 0.0
    best_checkpoint = None
    history = {"args": vars(args), "epochs": []}
    resume_path = os.path.join(args.out_dir, "resume.pt")
    history_path = os.path.join(args.out_dir, "history.json")
    last_pt_path = os.path.join(args.out_dir, "last.pt")
    start_epoch, start_batch, carried_loss = 0, 0, 0.0

    # 恢复优先级：1) resume.pt（训练中断） 2) last.pt + history.json（epoch完成）
    if os.path.exists(resume_path) and not args.no_resume:
        # 从训练中断点恢复
        st = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, start_batch, carried_loss = st["epoch"], st["batch"], st["epoch_loss"]
        best_val_f1, history = st["best_val_f1"], st["history"]
        if os.path.exists(os.path.join(args.out_dir, "best.pt")):
            best_checkpoint = os.path.join(args.out_dir, "best.pt")
        print(f"RESUMED from {resume_path}: epoch {start_epoch + 1}, micro-batch {start_batch}", flush=True)
        del st
    elif os.path.exists(history_path) and os.path.exists(last_pt_path) and not args.no_resume:
        # 从已完成的 epoch 继续
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        start_epoch = len(history["epochs"])
        if start_epoch < args.epochs:
            model.load_state_dict(torch.load(last_pt_path, map_location="cpu"))
            # 调整 scheduler：跳过已完成的 epoch 对应的步数
            completed_steps = start_epoch * (batches_per_epoch // args.grad_accum)
            for _ in range(completed_steps):
                scheduler.step()
            print(f"RESUMED from {last_pt_path}: continuing from epoch {start_epoch + 1} (scheduler advanced {completed_steps} steps)", flush=True)
            # 从 history 中恢复 best_val_f1
            if history["epochs"]:
                best_val_f1 = max(e["micro_f1"] for e in history["epochs"])
                if os.path.exists(os.path.join(args.out_dir, "best.pt")):
                    best_checkpoint = os.path.join(args.out_dir, "best.pt")
        else:
            print(f"Training already completed ({start_epoch}/{args.epochs} epochs), exiting.", flush=True)
            return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        sb = start_batch if epoch == start_epoch else 0

        def on_save(done, loss_so_far, _epoch=epoch):
            save_state_atomic(resume_path, {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": _epoch, "batch": done, "epoch_loss": loss_so_far,
                "best_val_f1": best_val_f1, "history": history})
            print(f"  [state saved at epoch {_epoch + 1}, micro-batch {done}]", flush=True)

        train_loss = train_one_epoch(
            model, make_train_loader(epoch, sb), optimizer, scheduler, scaler, device, args.grad_accum,
            args.grad_clip, class_weight, start_batch=sb, n_total=batches_per_epoch,
            total_loss=carried_loss if epoch == start_epoch else 0.0,
            save_every=args.save_every, on_save=on_save)
        val_loss, val_metrics = validate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={elapsed:.1f}s")
        print(f"  Val micro-F1={val_metrics['micro_f1']:.4f} | macro-F1={val_metrics['macro_f1']:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        history["epochs"].append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                                  "seconds": elapsed, **{k: float(v) for k, v in val_metrics.items()}})
        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_checkpoint = os.path.join(args.out_dir, "best.pt")
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  ✓ Saved best checkpoint to {best_checkpoint}")
        # 每个 epoch 完成后删除 resume.pt，避免下个 epoch 错误恢复
        if os.path.exists(resume_path):
            os.remove(resume_path)
        # 重置 start_batch 和 carried_loss，确保下个 epoch 从头开始
        start_batch = 0
        carried_loss = 0.0

    if os.path.exists(resume_path):
        os.remove(resume_path)  # run completed; a stale state must not be resumed later
    print(f"\nTraining finished. Best val micro-F1={best_val_f1:.4f}, checkpoint={best_checkpoint}")


if __name__ == "__main__":
    main()
