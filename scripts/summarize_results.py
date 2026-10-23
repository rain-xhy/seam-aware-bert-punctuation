"""汇总实验结果，生成表格和摘要。

    python -m scripts.summarize_results --results_dir results --out summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_all_results(results_dir: Path) -> Dict[str, dict]:
    """加载所有JSON结果文件"""
    results = {}
    for f in results_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            results[f.stem] = json.load(fp)
    return results


def format_table_row(cells: List[str], widths: List[int]) -> str:
    """格式化表格行"""
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"


def generate_methods_comparison_table(data: dict) -> str:
    """生成方法对比表"""
    if "methods" not in data:
        return ""

    methods = data["methods"]
    lines = []
    lines.append("### 方法对比表\n")

    # 表头
    widths = [20, 10, 10, 10, 12]
    lines.append(format_table_row(["方法", "F1", "TER", "耗时(s)", "吞吐(字/秒)"], widths))
    lines.append(format_table_row(["-" * w for w in widths], widths))

    # 数据行
    for name, stats in methods.items():
        f1 = f"{stats['quality']['micro_f1']:.4f}"
        ter = f"{stats['cost']['ter']:.3f}"
        time_s = f"{stats['cost']['seconds']:.1f}"
        throughput = f"{stats['cost']['chars_per_second']:.0f}"
        lines.append(format_table_row([name, f1, ter, time_s, throughput], widths))

    return "\n".join(lines)


def generate_edep_summary(data: dict) -> str:
    """生成EDEP摘要"""
    if "edep" not in data:
        return ""

    edep = data["edep"]
    lines = []
    lines.append("### EDEP边缘误差分析\n")
    lines.append(f"- **左边缘掩码宽度 (m_L)**: {edep['m_l']} 字符")
    lines.append(f"- **右边缘掩码宽度 (m_R)**: {edep['m_r']} 字符")
    lines.append(f"- **中心区域错误率**: {edep['centre_error']:.4f}")
    lines.append(f"- **窗口配置**: W={edep['window']}, V={edep['overlap']}, tail={edep['tail']}")

    return "\n".join(lines)


def generate_bootstrap_summary(data: dict) -> str:
    """生成Bootstrap检验摘要"""
    if "consistency" not in data and "delta" not in data:
        return ""

    lines = []
    lines.append("### 统计检验\n")

    if "consistency" in data:
        lines.append("**一致性检查:**")
        cons = data["consistency"]
        for key, val in cons.items():
            agreement = val.get("agreement", 0) * 100
            lines.append(f"- {key}: {agreement:.2f}% 一致")

    if "delta" in data:
        lines.append("\n**配对Bootstrap检验:**")
        lines.append(f"- F1差值: {data['delta']:.4f}")
        lines.append(f"- 95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]")
        lines.append(f"- 显著优于: {data['superior']}")

    return "\n".join(lines)


def generate_calibration_summary(data: dict) -> str:
    """生成温度校准摘要"""
    if "calibration" not in data:
        return ""

    lines = []
    lines.append("### 温度校准\n")
    cal = data["calibration"]
    lines.append(f"- **温度T**: {cal['temperature']:.4f}")
    lines.append(f"- **校准集**: {cal['split']}, {cal['n_samples']} 样本\n")

    lines.append("**效果对比:**")
    uncal = data["uncalibrated"]
    cal_res = data["calibrated"]
    imp = data["improvement"]

    lines.append(f"- 未校准: F1={uncal['micro_f1']:.4f}")
    lines.append(f"- 已校准: F1={cal_res['micro_f1']:.4f} (Δ{imp['micro_f1']:+.4f})")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="summary.md")
    args = ap.parse_args()

    results = load_all_results(Path(args.results_dir))

    lines = []
    lines.append("# 实验结果汇总\n")
    lines.append(f"共加载 {len(results)} 个结果文件\n")

    # E4: 方法对比
    if "E4_dev" in results:
        lines.append("## E4: 方法对比\n")
        lines.append(generate_methods_comparison_table(results["E4_dev"]))
        lines.append("")

    # E3: EDEP扫描
    if "E3_dev" in results:
        lines.append("## E3: 窗口扫描 + EDEP\n")
        lines.append(generate_edep_summary(results["E3_dev"]))
        lines.append("")

    # Bootstrap检验
    for key in results:
        if "bootstrap" in key.lower():
            lines.append("## Bootstrap显著性检验\n")
            lines.append(generate_bootstrap_summary(results[key]))
            lines.append("")
            break

    # 温度校准
    if "calibration" in results:
        lines.append("## 温度校准\n")
        lines.append(generate_calibration_summary(results["calibration"]))
        lines.append("")

    # Batch探测
    if "batch_probe" in results:
        data = results["batch_probe"]
        lines.append("## Batch探测\n")
        lines.append(f"推荐batch size: **{data['recommended_batch']}**\n")

    # 保存
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"结果摘要已保存到: {out_path}")


if __name__ == "__main__":
    main()
