"""科研级结果可视化
生成符合学术论文标准的高质量图表
"""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple

# 设置科研级绘图风格
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# 配色方案（色盲友好）
COLORS = {
    'primary': '#1f77b4',    # 蓝色
    'secondary': '#ff7f0e',  # 橙色
    'success': '#2ca02c',    # 绿色
    'warning': '#d62728',    # 红色
    'info': '#9467bd',       # 紫色
    'accent': '#8c564b',     # 棕色
    'neutral': '#7f7f7f',    # 灰色
    'highlight': '#e377c2',  # 粉色
}

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


class ResultVisualizer:
    """结果可视化器"""

    def __init__(self, results_dir: str = "results", output_dir: str = "figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_training_results(self) -> List[Dict]:
        """加载训练结果"""
        summary_file = self.results_dir / "training_summary.json"
        if not summary_file.exists():
            print(f"⚠️  训练汇总文件不存在: {summary_file}")
            return []

        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('models', [])

    def load_eval_results(self) -> Dict:
        """加载评估结果"""
        results = {}
        for json_file in self.results_dir.glob("E*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                results[json_file.stem] = json.load(f)
        return results

    def plot_training_comparison(self, models: List[Dict]):
        """绘制训练模型对比图"""
        if not models:
            print("⚠️  没有训练结果可绘制")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        names = [m['name'].replace('_', '\n') for m in models]
        micro_f1 = [m['best_f1'] for m in models]
        macro_f1 = [m['macro_f1'] for m in models]
        val_loss = [m['best_val_loss'] for m in models]

        # 按F1排序
        sorted_idx = np.argsort(micro_f1)[::-1]
        names = [names[i] for i in sorted_idx]
        micro_f1 = [micro_f1[i] for i in sorted_idx]
        macro_f1 = [macro_f1[i] for i in sorted_idx]
        val_loss = [val_loss[i] for i in sorted_idx]

        colors = [COLORS['primary'] if i == 0 else COLORS['neutral'] for i in range(len(names))]

        # Micro-F1
        ax = axes[0]
        bars = ax.barh(names, micro_f1, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Micro-F1', fontweight='bold')
        ax.set_title('(a) Micro-F1 Score', fontweight='bold')
        ax.set_xlim([min(micro_f1) - 0.005, max(micro_f1) + 0.005])
        for i, (bar, val) in enumerate(zip(bars, micro_f1)):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=8, fontweight='bold')

        # Macro-F1
        ax = axes[1]
        bars = ax.barh(names, macro_f1, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Macro-F1', fontweight='bold')
        ax.set_title('(b) Macro-F1 Score', fontweight='bold')
        ax.set_xlim([min(macro_f1) - 0.005, max(macro_f1) + 0.005])
        for i, (bar, val) in enumerate(zip(bars, macro_f1)):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=8)

        # Validation Loss
        ax = axes[2]
        bars = ax.barh(names, val_loss, color=[COLORS['warning'] if v == min(val_loss) else COLORS['neutral'] for v in val_loss],
                      edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Validation Loss', fontweight='bold')
        ax.set_title('(c) Validation Loss', fontweight='bold')
        ax.set_xlim([0, max(val_loss) * 1.1])
        for i, (bar, val) in enumerate(zip(bars, val_loss)):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=8)

        plt.suptitle('Training Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "training_comparison.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.savefig(self.output_dir / "training_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ 训练对比图已保存: {output_path}")
        plt.close()

    def plot_f1_vs_loss_scatter(self, models: List[Dict]):
        """绘制 F1 vs Loss 散点图"""
        if not models:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        micro_f1 = [m['best_f1'] for m in models]
        val_loss = [m['best_val_loss'] for m in models]
        names = [m['name'] for m in models]

        # 绘制散点
        scatter = ax.scatter(val_loss, micro_f1,
                           s=200, alpha=0.7,
                           c=range(len(models)),
                           cmap='viridis',
                           edgecolors='black', linewidth=1.5)

        # 添加标签
        for i, (x, y, name) in enumerate(zip(val_loss, micro_f1, names)):
            ax.annotate(name.replace('_', '\n'), (x, y),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

        # Pareto 前沿
        pareto_points = []
        for i, (x, y) in enumerate(zip(val_loss, micro_f1)):
            is_pareto = True
            for j, (x2, y2) in enumerate(zip(val_loss, micro_f1)):
                if i != j and x2 <= x and y2 >= y and (x2 < x or y2 > y):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append((x, y, i))

        if len(pareto_points) > 1:
            pareto_points.sort()
            pareto_x, pareto_y = zip(*[(p[0], p[1]) for p in pareto_points])
            ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
            ax.legend(loc='best')

        ax.set_xlabel('Validation Loss', fontweight='bold', fontsize=12)
        ax.set_ylabel('Micro-F1 Score', fontweight='bold', fontsize=12)
        ax.set_title('Quality-Loss Trade-off', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "f1_vs_loss.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(self.output_dir / "f1_vs_loss.png", dpi=300)
        print(f"✅ F1-Loss散点图已保存: {output_path}")
        plt.close()

    def plot_training_curves(self):
        """绘制训练曲线（从history.json）"""
        checkpoints_dir = Path("checkpoints")

        # 找到所有有history的模型
        models_with_history = []
        for model_dir in checkpoints_dir.glob("T*"):
            history_file = model_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                if history.get('epochs'):
                    models_with_history.append({
                        'name': model_dir.name,
                        'history': history
                    })

        if not models_with_history:
            print("⚠️  没有训练历史数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for i, model_data in enumerate(models_with_history):
            name = model_data['name']
            epochs_data = model_data['history']['epochs']

            epochs = [e['epoch'] for e in epochs_data]
            train_loss = [e['train_loss'] for e in epochs_data]
            val_loss = [e['val_loss'] for e in epochs_data]
            micro_f1 = [e['micro_f1'] for e in epochs_data]
            macro_f1 = [e['macro_f1'] for e in epochs_data]

            color = list(COLORS.values())[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]

            # Training Loss
            axes[0, 0].plot(epochs, train_loss, marker=marker, color=color,
                          label=name, linewidth=2, markersize=8)

            # Validation Loss
            axes[0, 1].plot(epochs, val_loss, marker=marker, color=color,
                          label=name, linewidth=2, markersize=8)

            # Micro-F1
            axes[1, 0].plot(epochs, micro_f1, marker=marker, color=color,
                          label=name, linewidth=2, markersize=8)

            # Macro-F1
            axes[1, 1].plot(epochs, macro_f1, marker=marker, color=color,
                          label=name, linewidth=2, markersize=8)

        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Training Loss', fontweight='bold')
        axes[0, 0].set_title('(a) Training Loss', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Validation Loss', fontweight='bold')
        axes[0, 1].set_title('(b) Validation Loss', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Micro-F1', fontweight='bold')
        axes[1, 0].set_title('(c) Micro-F1 Score', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Macro-F1', fontweight='bold')
        axes[1, 1].set_title('(d) Macro-F1 Score', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Training Curves Across Epochs', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "training_curves.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(self.output_dir / "training_curves.png", dpi=300)
        print(f"✅ 训练曲线已保存: {output_path}")
        plt.close()

    def plot_method_comparison_table(self, eval_results: Dict):
        """绘制方法对比表格（当评估结果可用时）"""
        # 这个函数在评估完成后调用
        pass

    def generate_all_figures(self):
        """生成所有图表"""
        print("="*60)
        print("生成科研级可视化图表")
        print("="*60)

        # 加载数据
        training_models = self.load_training_results()
        eval_results = self.load_eval_results()

        # 生成图表
        print("\n📊 生成训练对比图...")
        self.plot_training_comparison(training_models)

        print("\n📊 生成F1-Loss散点图...")
        self.plot_f1_vs_loss_scatter(training_models)

        print("\n📊 生成训练曲线...")
        self.plot_training_curves()

        print("\n" + "="*60)
        print(f"✅ 所有图表已保存到: {self.output_dir}/")
        print("="*60)


if __name__ == "__main__":
    visualizer = ResultVisualizer()
    visualizer.generate_all_figures()
