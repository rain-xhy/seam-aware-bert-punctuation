"""Launch one full training run (T1-T5, per 实现计划.md §3.1). Usage:

    python -m scripts.run_train T1
    python -m scripts.run_train T2   (fixed L=256 control)
    python -m scripts.run_train T3   (class-weighted loss control)
    python -m scripts.run_train T4   (center-only supervision control)
    python -m scripts.run_train T5   (no edge-aware shortening control)

Checkpoints go to checkpoints/<name>/, logs to logs/train_<name>.json.
GPU-only is assumed (raises if CUDA is unavailable, rather than silently
falling back to a CPU run that would take far longer than planned).
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys

import torch

from punct.char_codec import CharCodec
from punct.dataset import CropConfig
from punct.model import load_model
from punct.train import TrainConfig, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = {
    "T1": TrainConfig(crop=CropConfig(min_len=32, max_len=510, edge_aware_prob=0.3), micro_batch=8, grad_accum=2),
    "T2": TrainConfig(crop=CropConfig(min_len=256, max_len=256, edge_aware_prob=0.0), micro_batch=12, grad_accum=2),
    "T3": TrainConfig(crop=CropConfig(min_len=32, max_len=510, edge_aware_prob=0.3), class_weighted_loss=True,
                       micro_batch=8, grad_accum=2),
    "T4": TrainConfig(crop=CropConfig(min_len=32, max_len=510, edge_aware_prob=0.3, center_only_supervision=True,
                                       center_margin=50), micro_batch=8, grad_accum=2),
    "T5": TrainConfig(crop=CropConfig(min_len=32, max_len=510, edge_aware_prob=0.0), micro_batch=8, grad_accum=2),
}


def main(name: str):
    if name not in CONFIGS:
        raise ValueError(f"unknown run {name!r}; choices are {sorted(CONFIGS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run a full training job on CPU")

    device = torch.device("cuda")
    codec = CharCodec(os.path.join(ROOT, "bert-base-chinese", "vocab.txt"))
    model = load_model(os.path.join(ROOT, "bert-base-chinese"))
    cfg = CONFIGS[name]

    log = train(
        model,
        os.path.join(ROOT, "data", "train.npz"),
        os.path.join(ROOT, "data", "val.npz"),
        codec, cfg,
        out_dir=os.path.join(ROOT, "checkpoints", name),
        log_path=os.path.join(ROOT, "logs", f"train_{name}.json"),
        device=device,
    )
    print(f"{name} done: {len(log['epochs'])} epochs, "
          f"best val F1 = {max(e['val_micro_f1'] for e in log['epochs']):.4f}, "
          f"wall = {log['total_wall_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1])
