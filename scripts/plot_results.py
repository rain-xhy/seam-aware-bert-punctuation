"""生成实验结果图表。

读取results/目录下的JSON文件，生成：
1. F1-TER权衡图 (Pareto前沿)
2. EDEP-L/R曲线图
3. 训练曲线图
4. Batch探测曲线图

    python -m scripts.plot_results --results_dir results --out_dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def plot_f1_ter_tradeoff(data: dict, out_path: Path):
    """F1-TER权衡图 (Pareto前沿)"""
    if "points" not in data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for point in data["points"]:
        f1 = point["quality"]["micro_f1"]
        ter = point["cost"]["ter"]
        config = point["config"]
        label = f"W={config['window']},V={config['overlap']},{config['tail']}"
        ax.scatter(ter, f1, s=60, alpha=0.7)
        ax.annotate(label, (ter, f1), fontsize=8, alpha=0.6)

    ax.set_xlabel("TER (Token Expansion Ratio)", fontsize=12)
    ax.set_ylabel("Micro F1", fontsize=12)
    ax.set_title("F1 vs TER Trade-off", fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_edep_curves(data: dict, out_path: Path):
    """EDEP-L/R曲线图"""
    if "edep" not in data:
        return

    edep = data["edep"]
    distances = np.arange(len(edep["left"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(distances, edep["left"], 'b-', linewidth=2, label="EDEP-L (left edge)")
    ax.plot(distances, edep["right"], 'r-', linewidth=2, label="EDEP-R (right edge)")
    ax.axhline(y=edep["centre_error"], color='gray', linestyle='--',
               label=f"Centre baseline ({edep['centre_error']:.3f})")

    # 标记掩码阈值
    ax.axvline(x=edep["m_l"], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=edep["m_r"], color='red', linestyle=':', alpha=0.5)
    ax.text(edep["m_l"], ax.get_ylim()[1]*0.95, f"m_L={edep['m_l']}",
            fontsize=10, color='blue')
    ax.text(edep["m_r"], ax.get_ylim()[1]*0.90, f"m_R={edep['m_r']}",
            fontsize=10, color='red')

    ax.set_xlabel("Distance from edge (characters)", fontsize=12)
    ax.set_ylabel("Error rate", fontsize=12)
    ax.set_title("EDEP: Edge Distance Error Profile", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_training_curve(log_path: Path, out_path: Path):
    """训练曲线图"""
    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    except FileNotFoundError:
        return

    if "epochs" not in log:
        return

    epochs = [e["epoch"] for e in log["epochs"]]
    train_loss = [e["mean_train_loss"] for e in log["epochs"]]
    val_f1 = [e["val_micro_f1"] for e in log["epochs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss曲线
    ax1.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=14)
    ax1.grid(alpha=0.3)

    # F1曲线
    ax2.plot(epochs, val_f1, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Micro F1", fontsize=12)
    ax2.set_title("Validation F1", fontsize=14)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_batch_probe(data: dict, out_path: Path):
    """Batch探测曲线图"""
    if "results" not in data:
        return

    results = [r for r in data["results"] if r.get("success")]
    if not results:
        return

    batch_sizes = [r["batch_size"] for r in results]
    throughputs = [r["throughput"] for r in results]
    peak_mems = [r["peak_mem_mb"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 吞吐量
    ax1.plot(batch_sizes, throughputs, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Throughput (chars/s)", fontsize=12)
    ax1.set_title("Throughput vs Batch Size", fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.grid(alpha=0.3)

    # 显存
    ax2.plot(batch_sizes, peak_mems, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Peak Memory (MiB)", fontsize=12)
    ax2.set_title("Memory vs Batch Size", fontsize=14)
    ax2.set_xscale('log', base=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--log_file", help="training log JSON for curve")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("生成实验图表...")

    # F1-TER权衡图
    for f in results_dir.glob("E3*.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        plot_f1_ter_tradeoff(data, out_dir / f"f1_ter_{f.stem}.png")
        plot_edep_curves(data, out_dir / f"edep_{f.stem}.png")

    # Batch探测
    for f in results_dir.glob("batch_probe*.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        plot_batch_probe(data, out_dir / f"batch_probe_{f.stem}.png")

    # 训练曲线
    if args.log_file:
        plot_training_curve(Path(args.log_file), out_dir / "training_curve.png")

    print(f"\n所有图表已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
