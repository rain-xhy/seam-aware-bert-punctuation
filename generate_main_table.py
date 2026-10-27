"""生成方案要求的主表和权衡图
主表：所有方法的 P/R/F1、接缝F1、TER、耗时、加速比等
权衡图：精度-算力 Pareto 前沿
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List

# 科研级绘图设置
mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

COLORS = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
          'warning': '#d62728', 'info': '#9467bd', 'neutral': '#7f7f7f'}

def load_idle_timing(results_dir="results"):
    """T1 在测试集上、GPU 空闲时重复测得的耗时（results/timing/T1_test_rep*.json）。
    返回 {方法: {median, min, max, reps, chars_per_sec}}，没有重复文件时返回空字典。"""
    files = sorted(Path(results_dir, "timing").glob("T1_test_rep*.json"))
    if not files:
        return {}
    runs = [json.load(open(p, encoding='utf-8'))['methods'] for p in files]
    out = {}
    for name in runs[0]:
        secs = [r[name]['cost']['seconds'] for r in runs]
        cps = [r[name]['cost']['chars_per_second'] for r in runs]
        out[name] = {'median': float(np.median(secs)), 'min': float(min(secs)), 'max': float(max(secs)),
                     'reps': len(secs), 'chars_per_sec': float(np.median(cps))}
    return out


class MainTableGenerator:
    """生成方案§7要求的主表"""

    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)

    def load_e4_test(self):
        """加载E4测试集结果"""
        with open(self.results_dir / "E4_test.json", encoding='utf-8') as f:
            return json.load(f)

    def load_e7(self):
        """加载E7结果"""
        with open(self.results_dir / "E7_dev_full.json", encoding='utf-8') as f:
            return json.load(f)

    def generate_main_table(self):
        """生成主表"""
        print("="*80)
        print("主表：方法对比")
        print("="*80)

        e4 = self.load_e4_test()

        # 表头
        header = ["方法", "Precision", "Recall", "F1", "接缝F1", "TER", "耗时(s)", "字/秒", "相对串行加速"]
        print(f"\n{'方法':<20} {'P':>8} {'R':>8} {'F1':>8} {'接缝F1':>8} {'TER':>8} {'耗时':>8} {'字/秒':>10} {'加速':>8}")
        print("-"*100)

        methods = e4['methods']
        idle = load_idle_timing(self.results_dir)
        # 优先使用 GPU 空闲时的重复测量（中位数）；没有时退回 E4_test 里的单次测量
        sec = {n: (idle[n]['median'] if n in idle else d['cost']['seconds']) for n, d in methods.items()}
        serial_time = sec['serial_teacher']

        rows = []
        for name, data in methods.items():
            q = data['quality']
            c = dict(data['cost'])
            c['seconds'] = sec[name]
            if name in idle:
                c['chars_per_second'] = idle[name]['chars_per_sec']

            # 接缝F1（如果有seam_fixed）
            seam_f1 = q.get('seam_fixed', {}).get('seam_f1', 0.0)

            # 串行方法的 cost 里没有 TER（记为 None，表中显示 "-"，不当作 0 参与作图）
            ter = c.get('ter', data.get('config', {}).get('ter'))

            speedup = serial_time / c['seconds'] if c['seconds'] > 0 else 0

            row = {
                'method': name,
                'precision': q['micro_precision'],
                'recall': q['micro_recall'],
                'f1': q['micro_f1'],
                'seam_f1': seam_f1,
                'ter': ter,
                'seconds': c['seconds'],
                'chars_per_sec': c.get('chars_per_second', 0),
                'speedup': speedup
            }
            rows.append(row)

            ter_s = f"{row['ter']:>8.3f}" if row['ter'] is not None else f"{'-':>8}"
            print(f"{name:<20} {row['precision']:>8.4f} {row['recall']:>8.4f} {row['f1']:>8.4f} "
                  f"{row['seam_f1']:>8.4f} {ter_s} {row['seconds']:>8.1f} "
                  f"{row['chars_per_sec']:>10.0f} {row['speedup']:>8.2f}x")

        if idle:
            n_rep = next(iter(idle.values()))['reps']
            note = (f"耗时为 GPU 空闲时对 T1 在测试集上重复 {n_rep} 次的中位数；"
                    "加速比以课程串行规则 serial_teacher 为基准。")
            print(f"\n{note}")
            print("各方法三次重复的最小值与最大值：" +
                  "；".join(f"{n} {v['min']:.1f}-{v['max']:.1f}s" for n, v in idle.items()))
        else:
            note = "耗时来自单次运行，期间 GPU 可能被其他任务占用，仅供参考。"
            print(f"\n注意：{note}")

        # 保存为JSON
        with open(self.results_dir / "main_table.json", 'w', encoding='utf-8') as f:
            json.dump({'header': header, 'rows': rows, 'timing_note': note,
                       'timing_ranges': {n: [v['min'], v['max']] for n, v in idle.items()}},
                      f, ensure_ascii=False, indent=2)

        print("\n✅ 主表已保存: results/main_table.json")
        return rows

    def generate_control_table(self):
        """四个训练配置（主模型 + 三个对照）在测试集上、7 种解码方法下的 F1"""
        files = {
            'T1_varlen (主模型)': 'E4_test.json',
            'T3_noedge (无边缘感知)': 'T3_noedge_test.json',
            'T5_no_class_weight (无类别加权)': 'T5_no_class_weight_test.json',
            'T2_center_only (只监督中心区)': 'T2_center_only_test.json',
        }
        data = {}
        for label, fn in files.items():
            p = self.results_dir / fn
            if p.exists():
                with open(p, encoding='utf-8') as f:
                    data[label] = json.load(f)['methods']
            else:
                print(f"缺少 {fn}，跳过 {label}")
        if not data:
            return None

        method_names = list(next(iter(data.values())).keys())
        print("=" * 80)
        print("对照表：不同训练配置 × 解码方法（测试集 micro-F1）")
        print("=" * 80)
        print(f"{'方法':<16}" + "".join(f"{lab.split(' ')[0]:>16}" for lab in data))
        table = {}
        for m in method_names:
            table[m] = {lab: data[lab][m]['quality']['micro_f1'] for lab in data if m in data[lab]}
            print(f"{m:<16}" + "".join(f"{table[m].get(lab, float('nan')):>16.4f}" for lab in data))
        print("\n注意：本表只列点估计，模型间差距均小于 0.5 个百分点，不能直接解读为优劣；"
              "配对 bootstrap 区间见 results/control_bootstrap.json。")
        with open(self.results_dir / "control_table.json", 'w', encoding='utf-8') as f:
            json.dump({'metric': 'test micro_f1', 'table': table}, f, ensure_ascii=False, indent=2)
        print("\n✅ 对照表已保存: results/control_table.json")
        return table

class TradeoffPlotter:
    """绘制精度-算力权衡图（Pareto前沿）"""

    def __init__(self, results_dir="results", output_dir="figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_f1_vs_ter(self):
        """F1 vs TER 权衡图"""
        # 加载E3扫描数据
        with open(self.results_dir / "E3_dev.json", encoding='utf-8') as f:
            e3 = json.load(f)

        # 加载E4方法数据
        with open(self.results_dir / "E4_dev.json", encoding='utf-8') as f:
            e4 = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 6))

        # E3扫描点
        e3_points = []
        for p in e3.get('points', []):
            f1 = p['quality']['micro_f1']
            ter = p['cost']['ter']
            e3_points.append((ter, f1, p['config']))

        if e3_points:
            ters, f1s, _ = zip(*e3_points)
            ax.scatter(ters, f1s, s=30, alpha=0.5, c=COLORS['neutral'], label='E3 sweep points')

        # E4方法
        e4_methods = []
        skipped = []
        palette = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info'], COLORS['warning']]
        for name, data in e4['methods'].items():
            f1 = data['quality']['micro_f1']
            ter = data['cost'].get('ter', data.get('config', {}).get('ter'))
            if ter is None:  # 串行方法没有 TER，不画进 TER 轴
                skipped.append(name)
                continue
            color = palette[len(e4_methods) % len(palette)]
            e4_methods.append((ter, f1, name))
            ax.scatter([ter], [f1], s=150, alpha=0.9, marker='D', color=color, edgecolors='black',
                       linewidth=1.5, label=name, zorder=11)
            ax.annotate(name, (ter, f1), xytext=(8, 8), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        if skipped:
            ax.text(0.5, -0.16, 'Not shown (no TER recorded for serial methods): ' + ', '.join(skipped),
                    transform=ax.transAxes, ha='center', va='top', fontsize=8, color='gray')

        # Pareto前沿
        all_points = e3_points + e4_methods
        pareto = []
        for i, (ter, f1, cfg) in enumerate(all_points):
            is_pareto = True
            for j, (ter2, f12, _) in enumerate(all_points):
                if i != j and ter2 <= ter and f12 >= f1 and (ter2 < ter or f12 > f1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto.append((ter, f1))

        if len(pareto) > 1:
            pareto.sort()
            par_ter, par_f1 = zip(*pareto)
            ax.plot(par_ter, par_f1, 'r--', linewidth=2.5, alpha=0.7, label='Pareto Frontier', zorder=10)

        ax.set_xlabel('TER (Token Expansion Ratio)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Micro-F1 Score', fontweight='bold', fontsize=12)
        ax.set_title('Quality-Efficiency Trade-off (validation set)', fontweight='bold', fontsize=13)
        ax.legend(loc='center right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "pareto_f1_vs_ter.pdf", format='pdf')
        plt.savefig(self.output_dir / "pareto_f1_vs_ter.png", dpi=300)
        print(f"✅ Pareto图(F1 vs TER)已保存: {self.output_dir}/pareto_f1_vs_ter.pdf")
        plt.close()

    def plot_f1_vs_time(self):
        """F1 vs 实际耗时权衡图"""
        # F1 与耗时取自同一个集合：测试集。耗时是 GPU 空闲时重复测量的中位数。
        with open(self.results_dir / "E4_test.json", encoding='utf-8') as f:
            e4 = json.load(f)
        idle = load_idle_timing(self.results_dir)

        fig, ax = plt.subplots(figsize=(10, 6))

        points = []
        palette = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['info'],
                   COLORS['warning'], COLORS['neutral'], '#8c564b']
        for k, (name, data) in enumerate(e4['methods'].items()):
            f1 = data['quality']['micro_f1']
            time = idle[name]['median'] if name in idle else data['cost']['seconds']
            points.append((time, f1, name))
            ax.scatter([time], [f1], s=150, alpha=0.9, marker='o', color=palette[k % len(palette)],
                       edgecolors='black', linewidth=1.5, zorder=11)
            offsets = {'parallel': (-10, 22), 'serial_cross': (6, -26), 'serial_teacher': (10, 10),
                       'tang_parallel': (10, 10), 'direct_cut': (10, 10), 'serial_yu': (-70, 16)}
            ax.annotate(name, (time, f1), xytext=offsets.get(name, (8, 8)), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.6))

        # Pareto
        pareto = []
        for i, (t, f1, _) in enumerate(points):
            is_pareto = True
            for j, (t2, f12, _) in enumerate(points):
                if i != j and t2 <= t and f12 >= f1 and (t2 < t or f12 > f1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto.append((t, f1))

        if len(pareto) > 1:
            pareto.sort()
            pt, pf = zip(*pareto)
            ax.plot(pt, pf, 'r--', linewidth=2.5, alpha=0.7, label='Pareto Frontier', zorder=10)

        ax.set_xlabel('Wall-clock Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Micro-F1 Score', fontweight='bold', fontsize=12)
        ax.set_title('Quality-Speed Trade-off (test set)', fontweight='bold', fontsize=13)
        ax.set_xscale('log')
        ticks = [50, 100, 200, 500]
        ax.set_xlim(45, 700)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if idle:
            n_rep = next(iter(idle.values()))['reps']
            spread = max((v['max'] - v['min']) / v['median'] for v in idle.values()) * 100
            note = (f'Time: median of {n_rep} runs on an otherwise idle GPU (RTX 3050 4 GB, bf16); '
                    f'max-min spread at most {spread:.1f}% of the median.')
        else:
            note = 'Timings from a single run, GPU possibly shared.'
        ax.text(0.5, -0.16, note, transform=ax.transAxes, ha='center', va='top', fontsize=8, color='gray')
        if len(pareto) > 1:
            ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "pareto_f1_vs_time.pdf", format='pdf')
        plt.savefig(self.output_dir / "pareto_f1_vs_time.png", dpi=300)
        print(f"✅ Pareto图(F1 vs Time)已保存: {self.output_dir}/pareto_f1_vs_time.pdf")
        plt.close()

def main():
    print("\n" + "="*80)
    print("生成主表和权衡图")
    print("="*80 + "\n")

    # 主表
    table_gen = MainTableGenerator()
    table_gen.generate_main_table()

    print("\n")
    table_gen.generate_control_table()

    print("\n")

    # 权衡图
    plotter = TradeoffPlotter()
    plotter.plot_f1_vs_ter()
    plotter.plot_f1_vs_time()

    print("\n" + "="*80)
    print("✅ 主表和权衡图生成完成")
    print("="*80)

if __name__ == "__main__":
    main()
