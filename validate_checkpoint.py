"""验证已训练的检查点，补全 history.json"""
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification

# 导入项目模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from punct.data import BinnedWindowDataset
from punct.metrics import compute_metrics_from_2d

def validate(model, dataloader, device):
    """验证模型，返回 loss 和指标"""
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

            # 收集预测和标签
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, l in zip(preds, labels_np):
                mask = l != -100
                all_preds.extend(p[mask].tolist())
                all_labels.extend(l[mask].tolist())

    avg_loss = total_loss / len(dataloader)

    # 计算指标
    import numpy as np
    from collections import namedtuple
    EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])

    all_preds_arr = np.array(all_preds, dtype=np.int64).reshape(1, -1)
    all_labels_arr = np.array(all_labels, dtype=np.int64).reshape(1, -1)

    num_labels = 8
    logits_fake = np.zeros((1, all_preds_arr.shape[1], num_labels), dtype=np.float32)
    for i, p in enumerate(all_preds_arr[0]):
        logits_fake[0, i, p] = 1.0

    pred_obj = EvalPrediction(predictions=logits_fake, label_ids=all_labels_arr)
    metrics = compute_metrics_from_2d(pred_obj)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="检查点目录，例如 checkpoints/T4_fixed256")
    parser.add_argument("--epoch", type=int, required=True, help="这是第几个 epoch")
    parser.add_argument("--train_loss", type=float, required=True, help="训练 loss（从日志获取）")
    parser.add_argument("--train_time", type=float, required=True, help="训练时长（秒）")
    args = parser.parse_args()

    # 读取 history.json
    history_path = os.path.join(args.checkpoint_dir, "history.json")
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    # 加载模型
    bert_dir = history["args"]["bert_dir"]
    checkpoint_path = os.path.join(args.checkpoint_dir, "last.pt")

    print(f"Loading model from {checkpoint_path}...")
    model = BertForTokenClassification.from_pretrained(bert_dir, num_labels=8)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 加载验证数据
    data_dir = history["args"]["data_dir"]
    max_len = history["args"]["max_len"]
    min_len = history["args"].get("min_len", 32)
    edge_prob = history["args"]["edge_aware_prob"]
    center_only = history["args"]["center_only"]

    print(f"Loading validation data from {data_dir}...")
    val_ds = BinnedWindowDataset(
        data_dir=data_dir,
        split="val",
        min_window=min_len,
        max_window=max_len,
        edge_aware_prob=edge_prob,
        center_only=center_only,
        seed=42
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 验证
    print(f"Validating epoch {args.epoch}...")
    val_loss, val_metrics = validate(model, val_loader, device)

    print(f"\nEpoch {args.epoch} results:")
    print(f"  train_loss={args.train_loss:.4f}")
    print(f"  val_loss={val_loss:.4f}")
    print(f"  Val micro-F1={val_metrics['micro_f1']:.4f} | macro-F1={val_metrics['macro_f1']:.4f}")
    print(f"  time={args.train_time:.1f}s")

    # 更新 history.json
    epoch_record = {
        "epoch": args.epoch,
        "train_loss": args.train_loss,
        "val_loss": val_loss,
        "seconds": args.train_time,
        **{k: float(v) for k, v in val_metrics.items()}
    }
    history["epochs"].append(epoch_record)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Updated {history_path}")

    # 检查是否需要更新 best.pt
    if val_metrics["micro_f1"] > max((e["micro_f1"] for e in history["epochs"][:-1]), default=0):
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        import shutil
        shutil.copy2(checkpoint_path, best_path)
        print(f"✅ Saved new best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
