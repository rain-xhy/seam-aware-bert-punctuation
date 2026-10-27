"""继续未完成的训练（跳过已完成的 T3）"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

# T4: 完成1个epoch，从epoch 2继续（但没有resume.pt，会从头开始，依赖last.pt）
# T5: 未开始
# T2: 未开始

# 队列：T4, T5, T2
QUEUE = [
    ("T4_fixed256", ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
                     "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
                     "--out_dir", "checkpoints/T4_fixed256", "--min_len", "256", "--max_len", "256"]),
    ("T5_no_class_weight", ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
                            "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
                            "--out_dir", "checkpoints/T5_no_class_weight", "--max_len", "400"]),
    ("T2_center_only", ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
                        "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
                        "--out_dir", "checkpoints/T2_center_only", "--max_len", "400", "--center_only"])
]

print(f"继续训练队列: {' → '.join([n for n, _ in QUEUE])}")
print("=" * 60)

# 直接调用原脚本
sys.argv = [sys.argv[0]] + ["--skip-t3"]  # 标记参数（虽然不会用到，这里只是示意）

# 实际上直接导入并修改队列
from auto_train_all import TRAINING_QUEUE, main

# 修改队列：只保留 T4, T5, T2（完整参数）
TRAINING_QUEUE.clear()
TRAINING_QUEUE.extend([
    {"name": "T4_fixed256", "desc": "固定长度256对照",
     "args": ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
              "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
              "--out_dir", "checkpoints/T4_fixed256", "--min_len", "256", "--max_len", "256"]},
    {"name": "T5_no_class_weight", "desc": "无类别权重对照",
     "args": ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
              "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
              "--out_dir", "checkpoints/T5_no_class_weight", "--max_len", "400"]},
    {"name": "T2_center_only", "desc": "仅监督中心区对照",
     "args": ["--data_dir", "prepared_data_complete", "--bert_dir", "bert-base-chinese",
              "--epochs", "3", "--batch_size", "1", "--grad_accum", "8", "--seed", "42",
              "--out_dir", "checkpoints/T2_center_only", "--max_len", "400", "--center_only"]},
])

print(f"\n修改后队列: {[t['name'] for t in TRAINING_QUEUE]}\n")

# 运行
if __name__ == "__main__":
    main()
