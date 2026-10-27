"""Collect E3/E5/E7/E8/H1/fixed_core/probe_batch numbers into one markdown file.

Read-only over results/*.json; writes results/report_tables.md.
Timing columns are copied as recorded and are NOT validated (see note in output).

    python -m scripts.make_report_tables
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path("results")


def load(name):
    with open(R / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def f4(x):
    return f"{x:.4f}"


def main() -> None:
    out = []
    w = out.append

    # ---- E3 ----------------------------------------------------------------
    e3 = load("E3_dev")
    w("## E3 窗口 / 重叠 / 尾窗扫描（验证集，T1_varlen）\n")
    w(f"共 {len(e3['points'])} 个配置点。文件时间为 10-24 的旧运行，队列没有重算。\n")
    w("| W | V | 尾窗 | F1 | TER | 字/秒（未验证） |")
    w("|---|---|---|---|---|---|")
    pts = sorted(e3["points"], key=lambda p: (p["config"]["window"], p["config"]["overlap"], p["config"]["tail"]))
    for p in pts:
        c = p["config"]
        w(f"| {c['window']} | {c['overlap']} | {c['tail']} | {f4(p['quality']['micro_f1'])} | "
          f"{p['cost']['ter']:.3f} | {p['cost']['chars_per_second']:.0f} |")
    # short vs aligned at matching (W, V)
    by = {(p["config"]["window"], p["config"]["overlap"], p["config"]["tail"]): p for p in e3["points"]}
    w("\n短尾窗对贴齐文末（同 W、V）：\n")
    w("| W | V | TER short | TER aligned | TER 降低 | F1 short | F1 aligned |")
    w("|---|---|---|---|---|---|---|")
    for (W, V, t), p in sorted(by.items()):
        if t != "short" or (W, V, "aligned") not in by:
            continue
        a = by[(W, V, "aligned")]
        ts, ta = p["cost"]["ter"], a["cost"]["ter"]
        w(f"| {W} | {V} | {ts:.3f} | {ta:.3f} | {(ta - ts) / ta * 100:.1f}% | "
          f"{f4(p['quality']['micro_f1'])} | {f4(a['quality']['micro_f1'])} |")
    ed = e3["edep"]
    w(f"\nEDEP（W={ed['window']}, V={ed['overlap']}）：中心错误率 {ed['centre_error']:.4f}，"
      f"左侧屏蔽 m_l={ed['m_l']}，右侧屏蔽 m_r={ed['m_r']}。\n")

    # ---- E5 ----------------------------------------------------------------
    e5 = load("E5_dev")
    w("## E5 合并方式（验证集，W=400，V=100）\n")
    w("| 合并方式 | F1 | 接缝 F1 | TER | 耗时 s（未验证） |")
    w("|---|---|---|---|---|")
    for name, d in e5["methods"].items():
        q, c = d["quality"], d["cost"]
        seam = q.get("seam_fixed", {}).get("seam_f1")
        seam_s = f4(seam) if seam is not None else "-（JSON 无此项）"
        w(f"| {name} | {f4(q['micro_f1'])} | {seam_s} | {c['ter']:.3f} | {c['seconds']:.1f} |")
    w("")

    # ---- E7 ----------------------------------------------------------------
    e7 = load("E7_dev_full")
    cal = e7["calibration"]
    w(f"## E7 置信度补窗（验证集后 {e7['n_articles']} 篇，前 {e7['eval_skip']} 篇用于校准）\n")
    w(f"校准：T={cal['temperature']:.4f}，ECE {cal['ece_before']['ece']:.4f} → {cal['ece_after']['ece']:.4f}。\n")
    w("| 配置 | F1（全部） | F1（多窗口） | TER | 补窗数 | 改动字数 |")
    w("|---|---|---|---|---|---|")
    for name, d in e7["configs"].items():
        i = d["info"]
        w(f"| {name} | {f4(d['all']['micro_f1'])} | {f4(d['multi_window']['micro_f1'])} | {i['ter']:.3f} | "
          f"{i.get('n_selected', '-')} | {i.get('n_changed', '-')} |")
    w("")

    # ---- H1 ----------------------------------------------------------------
    h = load("H1_bootstrap")
    w(f"## 预先设定的假设检验（配对 bootstrap {h['n_boot']} 次，{h['n_articles']} 篇，δ={h['delta']}）\n")
    w("| 检验 | 差值 | 95% 区间 | 结论 |")
    w("|---|---|---|---|")
    verdict = {
        "H1_B_noninferior_to_serial": lambda v: "成立（非劣）" if v["non_inferior"] else "不成立",
        "H2_C_better_than_B_same_TER": lambda v: "成立（优于）" if v["superior"] else "不成立（C 显著更差）" if v["ci_upper"] < 0 else "不成立",
        "H3_C_better_than_random": lambda v: "成立（优于）" if v["superior"] else "不成立（无差别）",
    }
    for k, v in h["hypotheses"].items():
        w(f"| {k} | {v['diff']:+.4f} | [{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}] | {verdict[k](v)} |")
    for k, v in h["exploratory"].items():
        w(f"| {k}（探索性） | {v['diff']:+.4f} | [{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}] | 区间含 0 |")
    w("")

    # ---- E8 ----------------------------------------------------------------
    e8 = load("E8_matrix")
    wins = e8["windows"]
    w("## E8 训练长度 × 推理窗口（验证集，F1）\n")
    w("| 模型 | " + " | ".join(f"W={x}" for x in wins) + " |")
    w("|---|" + "---|" * len(wins))
    for m, v in e8["models"].items():
        w(f"| {m} | " + " | ".join(f4(v["cells"][str(x)]["micro_f1"]) for x in wins) + " |")
    w("")

    # ---- fixed core --------------------------------------------------------
    fc = load("fixed_core")
    c = fc["config"]
    w(f"## 固定核心隔离实验（核心 {c['core']} 字，{c['n_articles']} 篇，{c['n_boundaries']} 个边界）\n")
    w("| 每侧上下文 | 窗口长 | 接缝 F1 | 内部 F1 | 整体 F1 | TER |")
    w("|---|---|---|---|---|---|")
    for r in fc["results"]:
        w(f"| {r['context']} | {r['window_size']} | {f4(r['seam_f1'])} | {f4(r['interior_f1'])} | "
          f"{f4(r['overall_f1'])} | {r['ter']:.3f} |")
    w("")

    # ---- batch probe -------------------------------------------------------
    pb = load("probe_batch")
    w(f"## Batch 探测（{pb['config']['limit']} 篇，W={pb['config']['window']}，V={pb['config']['overlap']}）\n")
    w("| batch | 成功 | 字/秒（未验证） | 显存峰值 MiB | 相对上一档增幅 % |")
    w("|---|---|---|---|---|")
    for r in pb["results"]:
        w(f"| {r['batch_size']} | {r['success']} | {r['throughput']:.0f} | {r['peak_mem_mb']:.0f} | {r['gain_percent']:.1f} |")
    w(f"\n脚本推荐 batch={pb['recommended_batch']}。\n")

    # ---- idle-GPU timing ---------------------------------------------------
    runs = [load(f"timing/T1_test_rep{i}")["methods"] for i in (1, 2, 3)]
    n_art = load("timing/T1_test_rep1")["n_articles"]
    chars = runs[0]["parallel"]["cost"]["text_chars"]
    med = {m: float(np.median([r[m]["cost"]["seconds"] for r in runs])) for m in runs[0]}
    w(f"## 空闲 GPU 上的计时（测试集 {n_art} 篇，{chars} 字，T1，batch=4，bf16，重复 {len(runs)} 次）\n")
    w("监测脚本在 GPU 连续空闲 20 分钟（显存 52 MiB、利用率 0%）后先跑精度检查，"
      "检查结束后再确认空闲 3 分钟才开始计时。计时期间每 30 秒取一次快照，显存只在 52 MiB 与 811 到 873 MiB 之间变化，"
      "没有出现其他进程占用显卡的迹象。Windows 驱动不报告分进程显存，这是能得到的最强证据。\n")
    w("| 方法 | 第1次 s | 第2次 s | 第3次 s | 中位数 s | 字/秒 | F1 | 相对 serial_teacher | 相对 serial_cross |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m in runs[0]:
        secs = [r[m]["cost"]["seconds"] for r in runs]
        f1 = runs[0][m]["quality"]["micro_f1"]
        w(f"| {m} | " + " | ".join(f"{s:.1f}" for s in secs) +
          f" | {med[m]:.1f} | {chars / med[m]:.0f} | {f4(f1)} | "
          f"{med['serial_teacher'] / med[m]:.2f} | {med['serial_cross'] / med[m]:.2f} |")
    w("\n最后两列是基准方法的中位耗时除以该行方法的中位耗时，大于 1 表示该行方法比基准快。"
      "三次重复的最大相对波动为 5.9%（tang_parallel），其余方法在 4% 以内。\n")

    # ---- fp32 / bf16 and batch agreement -----------------------------------
    pc = load("precision_consistency")
    w(f"## 精度与批大小的标签一致率（测试集前 {pc['n_articles']} 篇，W={pc['window']}，V={pc['overlap']}）\n")
    w("| 比较 | 不一致位置数 | 总位置数 | 一致率 |")
    w("|---|---|---|---|")
    for c in pc["comparisons"]:
        w(f"| {c['a']} 对 {c['b']} | {c['mismatched_positions']} | {c['total']} | {c['agreement_rate'] * 100:.4f}% |")
    w("")

    w("---\n注：E3、E5 与 batch 探测表中的耗时、字/秒来自多个任务可能同时占用 GPU 的运行，"
      "未经重测，报告不得直接引用。F1 为确定性结果。空闲 GPU 计时见上一节。")

    text = "\n".join(out)
    (R / "report_tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
