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
    import seaborn as sns
except ImportError:
    print("Error: matplotlib/seaborn not installed. Run: pip install matplotlib seaborn")
    sys.exit(1)

LABEL_NAMES = ["无", "，", "。", "？", "！", "；", "：", "、"]  # same order as prepare_complete_data.LABELS


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


def plot_confusion_matrix(data: dict, out_path: Path, method_name: str = None):
    """混淆矩阵热力图"""
    confusion = None

    if "confusion" in data:
        confusion = np.array(data["confusion"])
    elif "methods" in data and method_name:
        confusion = np.array(data["methods"][method_name]["quality"].get("confusion"))

    if confusion is None or confusion.size == 0:
        return

    # 只显示7类标点（去掉第0类"无标点"）
    confusion_punct = confusion[1:, 1:]
    labels = LABEL_NAMES[1:]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(confusion_punct, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    title = f"混淆矩阵 - {method_name}" if method_name else "混淆矩阵"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_error_decomposition(data: dict, out_path: Path):
    """误差分解柱状图"""
    errors = data.get("errors")
    if not errors:
        return

    categories = ["漏标", "误插", "类别错"]
    values = [errors["missed"], errors["spurious"], errors["confused"]]
    total = sum(values)
    percentages = [v / total * 100 if total > 0 else 0 for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(categories, percentages, color=['#e74c3c', '#3498db', '#f39c12'], alpha=0.8)

    # 添加数值标签
    for bar, pct, val in zip(bars, percentages, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n({val})',
                ha='center', va='bottom', fontsize=11)

    ax.set_ylabel("错误占比 (%)", fontsize=12)
    ax.set_title("误差分解", fontsize=14)
    ax.set_ylim(0, max(percentages) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_methods_comparison(data: dict, out_path: Path):
    """多方法对比柱状图（F1 / TER / 吞吐）"""
    if "methods" not in data:
        return

    methods = data["methods"]
    names = list(methods.keys())
    f1s = [methods[n]["quality"]["micro_f1"] for n in names]
    ters = [methods[n]["cost"]["ter"] for n in names]
    throughputs = [methods[n]["cost"]["chars_per_second"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1对比
    axes[0].bar(range(len(names)), f1s, color='#2ecc71', alpha=0.8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel("Micro F1", fontsize=12)
    axes[0].set_title("质量对比 (F1)", fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)

    # TER对比
    axes[1].bar(range(len(names)), ters, color='#3498db', alpha=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel("TER", fontsize=12)
    axes[1].set_title("代价对比 (TER)", fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    # 吞吐对比
    axes[2].bar(range(len(names)), throughputs, color='#e74c3c', alpha=0.8)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel("字符/秒", fontsize=12)
    axes[2].set_title("效率对比 (吞吐)", fontsize=14)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_reliability(data: dict, out_path: Path):
    """Reliability diagram before/after temperature scaling (E7 calibration block)."""
    cal = data.get("calibration")
    if not cal or not isinstance(cal.get("ece_before"), dict):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title in ((axes[0], "ece_before", "before"), (axes[1], "ece_after", "after")):
        e = cal[key]
        min_count = max(1, int(0.001 * e["n"]))  # bins under 0.1% of samples are too noisy to plot
        bins = [b for b in e["bins"] if b["count"] >= min_count]
        dropped = sum(1 for b in e["bins"] if 0 < b["count"] < min_count)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
        for b in bins:
            ax.annotate(str(b["count"]), (b["confidence"], b["accuracy"]), textcoords="offset points",
                        xytext=(4, -10), fontsize=6)
        ax.set_title(f"{title} calibration, ECE={e['ece']:.4f}"
                     + (f"\n({dropped} sparse bin(s) hidden)" if dropped else ""), fontsize=10)
        ax.set_xlabel("confidence")
    axes[0].set_ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
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

    # F1-TER权衡图 & EDEP曲线
    for f in results_dir.glob("E3*.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        plot_f1_ter_tradeoff(data, out_dir / f"f1_ter_{f.stem}.png")
        plot_edep_curves(data, out_dir / f"edep_{f.stem}.png")

    # E7 校准可靠度图
    for f in results_dir.glob("E7*.json"):
        with open(f, encoding="utf-8") as fp:
            plot_reliability(json.load(fp), out_dir / f"reliability_{f.stem}.png")

    # E4方法对比
    for f in results_dir.glob("E4*.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        plot_methods_comparison(data, out_dir / f"methods_comparison_{f.stem}.png")

        # 第一个方法的混淆矩阵
        if "methods" in data:
            first_method = list(data["methods"].keys())[0]
            # 需要从完整结果中获取confusion，这里简化处理
            if "confusion" in data.get("methods", {}).get(first_method, {}).get("quality", {}):
                plot_confusion_matrix(data, out_dir / f"confusion_{f.stem}.png", first_method)

        # 误差分解（如果有）
        if "errors" in data:
            plot_error_decomposition(data, out_dir / f"errors_{f.stem}.png")

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
