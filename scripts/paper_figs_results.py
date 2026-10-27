"""Figures 2 to 6 of the paper, all drawn from files in results/ (nothing is re-run on the GPU).

    python -m scripts.paper_figs_results

fig2_time      test-set F1 against idle-GPU wall-clock time (median of three repetitions)
fig3_ter       validation-set F1 against TER for the 30 sweep configurations and three methods
fig4_fixedcore seam and interior F1 against the context added on each side of a fixed core
fig5_edep      edge-distance error profile, left and right edge
fig6_patch     second-pass patching: F1 against TER, and net corrected positions
"""
from __future__ import annotations

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_fig_common import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, save

R = "results"
LABEL = {"direct_cut": "直接截断", "serial_yu": "串行 W=64", "serial_tang": "串行 W=125",
         "serial_teacher": "课程串行 W=400", "serial_cross": "跨文章串行", "parallel": "并行滑窗",
         "tang_parallel": "Tang 并行"}
FAMILY = {"direct_cut": "直接截断", "serial_yu": "逐篇串行", "serial_tang": "逐篇串行",
          "serial_teacher": "逐篇串行", "serial_cross": "跨文章成批串行", "parallel": "并行滑窗",
          "tang_parallel": "并行滑窗"}
FCOLOR = {"逐篇串行": BLUE, "跨文章成批串行": AQUA, "并行滑窗": ORANGE, "直接截断": MUTED}


def load(name):
    with open(f"{R}/{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def fig_time():
    e4 = load("E4_test")["methods"]
    reps = [json.load(open(p, encoding="utf-8"))["methods"] for p in
            sorted(glob.glob(f"{R}/timing/T1_test_rep*.json"))]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    offs = {"direct_cut": (0, 13), "serial_yu": (-9, 10), "serial_tang": (9, 8), "serial_teacher": (9, 8),
            "serial_cross": (0, -15), "parallel": (0, 13), "tang_parallel": (0, -15)}
    for name, d in e4.items():
        t = float(np.median([r[name]["cost"]["seconds"] for r in reps]))
        lo = min(r[name]["cost"]["seconds"] for r in reps)
        hi = max(r[name]["cost"]["seconds"] for r in reps)
        f1 = d["quality"]["micro_f1"]
        c = FCOLOR[FAMILY[name]]
        ax.errorbar([t], [f1], xerr=[[t - lo], [hi - t]], fmt="none", ecolor=c, elinewidth=1.2, capsize=3, zorder=2)
        ax.scatter([t], [f1], s=70, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy = offs[name]
        ax.annotate(LABEL[name], (t, f1), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK2,
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center")
    for fam, c in FCOLOR.items():
        ax.scatter([], [], s=50, color=c, label=fam)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 400, 600])
    ax.set_xticklabels(["50", "100", "200", "400", "600"])
    ax.set_xlim(45, 760)
    ax.set_ylim(0.7785, 0.8095)
    ax.set_xlabel("墙钟耗时 s，对数轴，测试集 3238 篇，空闲显卡三次重复的中位数，横线为最小到最大")
    ax.set_ylabel("测试集 micro F1")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_time")


def pareto(points):
    out = []
    for i, (x, y) in enumerate(points):
        if not any(j != i and x2 <= x and y2 >= y and (x2 < x or y2 > y) for j, (x2, y2) in enumerate(points)):
            out.append((x, y))
    return sorted(out)


def fig_ter():
    e3 = load("E3_dev")["points"]
    e4 = load("E4_dev")["methods"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pts = [(p["cost"]["ter"], p["quality"]["micro_f1"]) for p in e3]
    short = [(p["cost"]["ter"], p["quality"]["micro_f1"]) for p in e3 if p["config"]["tail"] == "short"]
    aligned = [(p["cost"]["ter"], p["quality"]["micro_f1"]) for p in e3 if p["config"]["tail"] == "aligned"]
    ax.scatter(*zip(*aligned), s=26, facecolor="none", edgecolor=MUTED, linewidth=1.1, label="贴齐文末", zorder=2)
    ax.scatter(*zip(*short), s=26, color=BLUE, alpha=0.75, label="短尾窗", zorder=2)
    meth = {}
    for name in ("direct_cut", "parallel", "tang_parallel"):
        meth[name] = (e4[name]["cost"]["ter"], e4[name]["quality"]["micro_f1"])
    fr = pareto(pts + list(meth.values()))
    ax.step(*zip(*fr), where="post", color=INK2, linewidth=1.2, linestyle="--", label="前沿", zorder=1)
    cols = {"direct_cut": MUTED, "parallel": ORANGE, "tang_parallel": AQUA}
    offs = {"direct_cut": (10, 4), "parallel": (10, -14), "tang_parallel": (10, -4)}
    for name, (x, y) in meth.items():
        ax.scatter([x], [y], s=95, marker="D", color=cols[name], edgecolor="white", linewidth=1.3, zorder=4)
        ax.annotate(LABEL[name], (x, y), xytext=offs[name], textcoords="offset points", fontsize=9, color=INK2)
    best = max(e3, key=lambda p: p["quality"]["micro_f1"])
    ax.annotate("窗口 256 重叠 200", (best["cost"]["ter"], best["quality"]["micro_f1"]), xytext=(-8, -16),
                textcoords="offset points", fontsize=9, color=INK2, ha="right")
    ax.set_xlabel("前向字数比 TER，验证集")
    ax.set_ylabel("验证集 micro F1")
    ax.set_ylim(0.775, 0.812)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig3_ter")


def fig_fixedcore():
    fc = load("fixed_core")["results"]
    ctx = [r["context"] for r in fc]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(ctx))
    ax.plot(x, [r["seam_f1"] for r in fc], color=ORANGE, marker="o", markersize=7, linewidth=2,
            markeredgecolor="white", markeredgewidth=1.3)
    ax.plot(x, [r["interior_f1"] for r in fc], color=BLUE, marker="o", markersize=7, linewidth=2,
            markeredgecolor="white", markeredgewidth=1.3)
    for i, r in enumerate(fc):
        ax.annotate(f"{r['seam_f1']:.4f}", (i, r["seam_f1"]), xytext=(0 if i == 0 else 9, -17),
                    textcoords="offset points", ha="center" if i == 0 else "left", fontsize=9, color=INK2)
    ax.text(len(ctx) - 1 + 0.35, fc[-1]["seam_f1"] - 0.002, "接缝 F1", color=INK2, fontsize=9.5, va="center")
    ax.text(len(ctx) - 1 + 0.35, fc[-1]["interior_f1"] + 0.002, "内部 F1", color=INK2, fontsize=9.5, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['context']}\nTER {r['ter']:.3f}" for r in fc])
    ax.set_xlim(-0.3, len(ctx) - 1 + 1.0)
    ax.set_ylim(0.70, 0.82)
    ax.set_xlabel("每侧补充的上下文字数，核心 192 字，验证集 1559 篇")
    ax.set_ylabel("micro F1")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig4_fixedcore")


def fig_edep():
    ed = load("E3_dev")["edep"]
    d = np.arange(len(ed["left"]))
    base = ed["centre_error"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(d, ed["left"], color=BLUE, linewidth=1.6, label="左边缘")
    ax.plot(d, ed["right"], color=ORANGE, linewidth=1.6, label="右边缘")
    ax.axhline(base, color=INK2, linewidth=1.0)
    ax.axhline(base + 0.005, color=INK2, linewidth=1.0, linestyle=":")
    ax.text(102, base - 0.0006, f"基线 {base:.4f}", color=INK2, fontsize=9, va="top")
    ax.text(102, base + 0.005 + 0.0006, f"基线加容差 {base + 0.005:.4f}", color=INK2, fontsize=9, va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_xlim(0, 126)
    ax.set_xlabel("到窗口边缘的距离 字，窗口 510，重叠 255，验证集，每个距离 3747 个位置")
    ax.set_ylabel("错误率")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig5_edep")


def fig_patch():
    e7 = load("E7_dev_full")["configs"]
    ch = json.load(open(f"{R}/analysis_for_results.json", encoding="utf-8"))["e7_changes"]
    names = [("pass1", "只跑第一遍"), ("c_calibrated", "校准选位补窗"), ("c_uncalibrated", "未校准选位补窗"),
             ("c_random", "随机选位补窗"), ("b_same_ter", "加大重叠至同 TER")]
    cols = [MUTED, ORANGE, ORANGE, ORANGE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for (k, lab), c in zip(names, cols):
        x, y = e7[k]["info"]["ter"], e7[k]["all"]["micro_f1"]
        ax.scatter([x], [y], s=85, color=c, edgecolor="white", linewidth=1.4, zorder=3)
        ha, dx = ("right", -9) if k == "b_same_ter" else ("left", 9)
        if k == "c_uncalibrated":
            dx, ha, dy = 0, "center", 16
        elif k == "c_calibrated":
            dx, ha, dy = 0, "center", -16
        elif k == "pass1":
            dx, ha, dy = 0, "center", 16
        else:
            dy = 0
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK2,
                    ha=ha, va="center")
    ax.set_xlim(1.1, 3.0)
    ax.set_ylim(0.8026, 0.8056)
    ax.set_xlabel("前向字数比 TER")
    ax.set_ylabel("micro F1，验证集后 2676 篇")
    ax.set_title("a 质量与算力", loc="left", fontsize=10.5, color=INK)
    ax.grid(True, axis="y")
    ax = axes[1]
    labs = [lab for k, lab in names[1:]]
    net = [ch[k]["wrong_to_right"] - ch[k]["right_to_wrong"] for k, _ in names[1:]]
    y = np.arange(len(labs))[::-1]
    ax.barh(y, net, color=[ORANGE, ORANGE, ORANGE, BLUE], height=0.55)
    ax.axvline(0, color=INK2, linewidth=1.0)
    for yi, v in zip(y, net):
        ax.text(v + (8 if v >= 0 else -8), yi, f"{v:+d}", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlim(-60, 450)
    ax.set_xlabel("净改对位置数，改对减改错")
    ax.set_title("b 相对第一遍的净改动", loc="left", fontsize=10.5, color=INK)
    ax.grid(True, axis="x")
    fig.tight_layout()
    save(fig, "fig6_patch")


def main():
    fig_time()
    fig_ter()
    fig_fixedcore()
    fig_edep()
    fig_patch()


if __name__ == "__main__":
    main()
