"""Smoke test: 200 optimizer steps on real data, confirm loss goes down and
the encoder weights actually change. Not a real training run (checkpoint is
discarded). Usage:

    conda activate yolov8 (or set HF_HUB_OFFLINE etc. directly)
    python -m scripts.smoke_train
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tempfile

import torch

from punct.char_codec import CharCodec
from punct.dataset import CropConfig
from punct.model import load_model
from punct.train import TrainConfig, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec(os.path.join(ROOT, "bert-base-chinese", "vocab.txt"))
    model = load_model(os.path.join(ROOT, "bert-base-chinese"))

    cfg = TrainConfig(
        crop=CropConfig(min_len=32, max_len=510, edge_aware_prob=0.3),
        epochs=1,
        micro_batch=8,
        grad_accum=1,
        steps_per_epoch=200,  # smoke test: 200 optimizer steps, not a full epoch
        val_max_articles=50,  # smoke test: score 50 of 3223 val articles, not the whole split
    )

    with tempfile.TemporaryDirectory() as out_dir:
        log = train(
            model, os.path.join(ROOT, "data", "train.npz"), os.path.join(ROOT, "data", "val.npz"),
            codec, cfg, out_dir=out_dir, log_path=os.path.join(ROOT, "logs", "smoke_train.json"),
            device=device,
        )
    print("mean_train_loss (first vs last 20 steps would need per-step log; "
          "epoch mean reported):", log["epochs"][0]["mean_train_loss"])
    print("encoder_weights_changed:", log["encoder_weights_changed"],
          "max_abs_diff:", log["encoder_weight_max_abs_diff"])
    print("val_micro_f1 after 200 steps:", log["epochs"][0]["val_micro_f1"])


if __name__ == "__main__":
    main()
