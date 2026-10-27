"""评估所有训练好的模型在测试集上的性能"""
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

from punct.dataset import WindowDataset, CropConfig, collate_windows
from punct.metrics import compute_metrics
from punct.char_codec import CharCodec
import numpy as np


def validate(model, dataloader, device, split_name="test"):
    """在指定数据集上评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    print(f"  Evaluating on {split_name}...", flush=True)
    n_batches = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i > 0 and i % 1000 == 0:
                print(f"    batch {i}/{n_batches}", flush=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()

            # 收集预测和标签（按 compute_metrics 的要求）
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)

    # 计算指标 - 使用 compute_metrics
    from collections import namedtuple
    EvalPrediction = namedtuple('EvalPrediction', ['predictions', 'label_ids'])

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    pred_obj = EvalPrediction(predictions=all_preds_tensor.numpy(), label_ids=all_labels_tensor.numpy())
    metrics = compute_metrics(pred_obj)

    return avg_loss, metrics


# 模型配置
MODELS = [
    {
        "name": "T3_noedge",
        "desc": "无边缘感知（baseline）",
        "checkpoint": "checkpoints/T3_noedge/best.pt",
        "args": {
            "max_len": 400,
            "min_len": 32,
            "edge_aware_prob": 0.0,
            "center_only": False
        }
    },
    {
        "name": "T4_fixed256",
        "desc": "固定长度256对照",
        "checkpoint": "checkpoints/T4_fixed256/best.pt",
        "args": {
            "max_len": 256,
            "min_len": 256,
            "edge_aware_prob": 0.3,
            "center_only": False
        }
    },
    {
        "name": "T5_no_class_weight",
        "desc": "无类别权重对照",
        "checkpoint": "checkpoints/T5_no_class_weight/best.pt",
        "args": {
            "max_len": 400,
            "min_len": 32,
            "edge_aware_prob": 0.3,
            "center_only": False
        }
    },
    {
        "name": "T2_center_only",
        "desc": "仅监督中心区对照",
        "checkpoint": "checkpoints/T2_center_only/best.pt",
        "args": {
            "max_len": 400,
            "min_len": 32,
            "edge_aware_prob": 0.3,
            "center_only": True
        }
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="prepared_data_complete", help="数据目录")
    parser.add_argument("--bert_dir", default="bert-base-chinese", help="BERT模型目录")
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="评估数据集")
    parser.add_argument("--out", default="results/final_evaluation.json", help="输出文件")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 codec
    codec_path = os.path.join(args.data_dir, "char_codec.json")
    codec = CharCodec.load(codec_path)

    results = {}

    for model_config in MODELS:
        model_name = model_config["name"]
        checkpoint_path = model_config["checkpoint"]

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}: {model_config['desc']}")
        print(f"{'='*60}")

        if not os.path.exists(checkpoint_path):
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            results[model_name] = {"error": "checkpoint not found"}
            continue

        # 加载模型
        print(f"  Loading model from {checkpoint_path}...")
        model = BertForTokenClassification.from_pretrained(args.bert_dir, num_labels=8)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.to(device)

        # 配置数据集（测试时不使用边缘感知和center_only）
        crop_config = CropConfig(
            min_len=model_config["args"]["min_len"],
            max_len=model_config["args"]["max_len"],
            edge_aware_prob=0.0,  # 测试时不使用边缘感知
            center_only_supervision=False,  # 测试时全监督
            center_margin=64
        )

        # 加载数据
        print(f"  Loading {args.split} data...")
        test_ds = WindowDataset(
            data_dir=args.data_dir,
            split=args.split,
            cfg=crop_config,
            seed=42
        )

        def collate_fn(batch):
            return collate_windows(batch, codec.cls_id, codec.sep_id, codec.pad_id)

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

        # 评估
        test_loss, test_metrics = validate(model, test_loader, device, args.split)

        print(f"\n  Results:")
        print(f"    Loss: {test_loss:.4f}")
        print(f"    Micro-F1: {test_metrics['micro_f1']:.4f}")
        print(f"    Macro-F1: {test_metrics['macro_f1']:.4f}")

        # 保存结果
        results[model_name] = {
            "description": model_config["desc"],
            "checkpoint": checkpoint_path,
            "split": args.split,
            "loss": test_loss,
            **{k: float(v) for k, v in test_metrics.items()}
        }

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 保存结果到文件
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {args.out}")
    print(f"{'='*60}")

    # 打印排名
    print("\n📊 Final Ranking (by Micro-F1):")
    sorted_models = sorted(
        [(name, res.get("micro_f1", 0)) for name, res in results.items() if "micro_f1" in res],
        key=lambda x: x[1],
        reverse=True
    )
    for i, (name, f1) in enumerate(sorted_models, 1):
        medal = ["🥇", "🥈", "🥉", ""][min(i-1, 3)]
        print(f"  {medal} {i}. {name}: F1={f1:.4f}")


if __name__ == "__main__":
    main()
