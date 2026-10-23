"""Training loop: fine-tune bert-base-chinese for 8-class punctuation labelling.

完善方案.md §3.1 starting hyperparameters: AdamW, lr 3e-5, weight decay 0.01,
10% warmup then linear decay, grad-clip 1.0, 3 epochs, bf16 autocast,
micro-batch 8 + gradient accumulation 2. §3.2 training samples: variable-
length random crops (dataset.pick_crop), optionally with edge-aware
shortening and/or center-only supervision, depending on which of T1-T5 is
being trained (实现计划.md §3.1).

Model selection: best checkpoint by validation micro-F1 over the 7 pause-
mark classes (NONE excluded), not by validation loss. Every run writes a
JSON log (loss per step, val F1 per epoch, wall-clock, GPU peak memory,
encoder-weight-changed check) so the report can point every number back to
one of these files, per the阶段4 requirement.
"""
from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from punct.char_codec import NUM_LABELS
from punct.dataset import CropConfig, collate_batch, pick_crop
from punct.metrics import punctuation_prf

IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    crop: CropConfig
    class_weighted_loss: bool = False
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0
    epochs: int = 3
    micro_batch: int = 8
    grad_accum: int = 2
    seed: int = 0
    steps_per_epoch: Optional[int] = None  # None = one pass over all articles
    val_max_articles: Optional[int] = None  # None = evaluate on the full val set


def _steps_for_one_epoch(num_articles: int, micro_batch: int) -> int:
    return max(1, num_articles // micro_batch)


def _class_weights(label_ids: np.ndarray, num_labels: int = NUM_LABELS, cap: float = 8.0) -> torch.Tensor:
    """Inverse-sqrt-frequency weights, relative to class 0 (no punctuation),
    capped at `cap`. Used by T3 (完善方案.md §3.3)."""
    counts = np.bincount(label_ids, minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for absent classes
    inv_sqrt = 1.0 / np.sqrt(counts)
    weights = inv_sqrt / inv_sqrt[0]
    weights = np.minimum(weights, cap)
    return torch.tensor(weights, dtype=torch.float32)


def _sample_batch(
    char_ids: np.ndarray, label_ids: np.ndarray, boundaries: np.ndarray,
    cfg: CropConfig, rng: random.Random, batch_size: int, cls_id: int, sep_id: int, pad_id: int,
):
    n_articles = len(boundaries) - 1
    crops = []
    while len(crops) < batch_size:
        idx = rng.randrange(n_articles)
        start, end = boundaries[idx], boundaries[idx + 1]
        crop = pick_crop(char_ids[start:end], label_ids[start:end], idx, cfg, rng)
        if crop is not None:
            crops.append(crop)
    return collate_batch(crops, cls_id=cls_id, sep_id=sep_id, pad_id=pad_id)


def train(
    model, train_npz_path: str, val_npz_path: str, codec, cfg: TrainConfig,
    out_dir: str, log_path: str, device: torch.device,
) -> Dict:
    """Run the full training loop, save the best checkpoint to out_dir, write
    a JSON training log to log_path, and return that same log dict."""
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_data = np.load(train_npz_path, allow_pickle=True)
    val_data = np.load(val_npz_path, allow_pickle=True)
    train_char_ids, train_label_ids, train_bounds = (
        train_data["char_ids"], train_data["label_ids"].astype(np.int64), train_data["boundaries"]
    )
    n_articles = len(train_bounds) - 1
    steps_per_epoch = cfg.steps_per_epoch or _steps_for_one_epoch(n_articles, cfg.micro_batch)
    total_steps = steps_per_epoch * cfg.epochs // cfg.grad_accum

    weight_tensor = None
    if cfg.class_weighted_loss:
        weight_tensor = _class_weights(train_label_ids).to(device)

    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps
    )

    log: Dict = {
        "config": {**asdict(cfg), "crop": asdict(cfg.crop)},
        "n_train_articles": n_articles,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps_planned": total_steps,
        "epochs": [],
    }

    before_check = model.bert.encoder.layer[0].attention.self.query.weight.detach().clone().cpu()
    start_time = time.time()
    optimizer_updates = 0

    for epoch in range(cfg.epochs):
        epoch_losses: List[float] = []
        epoch_start = time.time()
        model.train()
        for micro_step in range(steps_per_epoch):
            input_ids, attention_mask, label_ids = _sample_batch(
                train_char_ids, train_label_ids, train_bounds, cfg.crop, rng,
                cfg.micro_batch, codec.cls_id, codec.sep_id, codec.pad_id,
            )
            input_ids_t = torch.from_numpy(input_ids).to(device)
            attention_mask_t = torch.from_numpy(attention_mask).to(device)
            label_ids_t = torch.from_numpy(label_ids).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(input_ids=input_ids_t, attention_mask=attention_mask_t)
                logits = out.logits
                if weight_tensor is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, NUM_LABELS), label_ids_t.view(-1),
                        weight=weight_tensor, ignore_index=IGNORE_INDEX,
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, NUM_LABELS), label_ids_t.view(-1), ignore_index=IGNORE_INDEX,
                    )
            (loss / cfg.grad_accum).backward()
            epoch_losses.append(loss.item())

            if (micro_step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1

        val_f1, val_report = evaluate_full_articles(model, val_data, codec, device, max_articles=cfg.val_max_articles)
        epoch_record = {
            "epoch": epoch + 1,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "val_micro_f1": val_f1,
            "val_report": val_report,
            "epoch_seconds": time.time() - epoch_start,
        }
        log["epochs"].append(epoch_record)
        print(f"epoch {epoch + 1}: loss={epoch_record['mean_train_loss']:.4f} val_F1={val_f1:.4f}")

        best_so_far = max(e["val_micro_f1"] for e in log["epochs"])
        if val_f1 == best_so_far:
            _save_checkpoint(model, out_dir)

    after_check = model.bert.encoder.layer[0].attention.self.query.weight.detach().clone().cpu()
    max_diff = float((after_check - before_check).abs().max().item())
    log["encoder_weight_max_abs_diff"] = max_diff
    log["encoder_weights_changed"] = max_diff > 1e-6
    log["total_wall_seconds"] = time.time() - start_time
    log["optimizer_updates_completed"] = optimizer_updates
    if device.type == "cuda":
        log["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)

    del train_data, val_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return log


def _save_checkpoint(model, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))


def evaluate_full_articles(model, npz_data, codec, device, max_window: int = 510,
                            max_articles: Optional[int] = None) -> tuple:
    """Evaluate on (optionally the first max_articles of) every article,
    truncated to max_window chars (a cheap, approximate validation signal
    during training; the real multi-window inference pipeline is evaluated
    separately in E0/E3/etc. via infer_parallel.py). Returns
    (micro_f1, full_report_dict). max_articles exists so smoke tests do not
    have to score the entire validation split just to see loss going down."""
    char_ids, label_ids, boundaries = npz_data["char_ids"], npz_data["label_ids"].astype(np.int64), npz_data["boundaries"]
    model.eval()
    all_preds: List[int] = []
    all_gold: List[int] = []
    n_articles = len(boundaries) - 1
    limit = n_articles if max_articles is None else min(max_articles, n_articles)
    with torch.no_grad():
        for i in range(limit):
            start, end = boundaries[i], min(boundaries[i] + max_window, boundaries[i + 1])
            if end <= start:
                continue
            ids = [codec.cls_id] + char_ids[start:end].tolist() + [codec.sep_id]
            input_ids = torch.tensor([ids], device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits[0, 1:-1].float().argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_gold.extend(label_ids[start:end].tolist())
    report = punctuation_prf(np.array(all_gold), np.array(all_preds), num_labels=NUM_LABELS)
    return report["micro_f1"], report
