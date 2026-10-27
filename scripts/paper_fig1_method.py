"""Figure 1: window planning (aligned vs short tail) and the two-pass patch scheme.

    python -m scripts.paper_fig1_method
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.paper_fig_common import AQUA, BLUE, INK, INK2, MUTED, ORANGE, save

N, W, V = 900, 400, 100
S = W - V


def windows(tail):
    starts = list(range(0, N - W + 1, S))
    ws = [(s, s + W) for s in starts]
    if ws[-1][1] < N:
        ws.append((N - W, N) if tail == "aligned" else (ws[-1][0] + S, N))
    return ws


def draw_row(ax, y, ws, color, label):
    for i, (s, e) in enumerate(ws):
        yy = y - (i % 2) * 0.42
        ax.add_patch(Rectangle((s, yy), e - s, 0.32, fc=color, ec="white", lw=1.5, alpha=0.85))
        ax.text((s + e) / 2, yy + 0.16, f"{e - s}", ha="center", va="center", color="white", fontsize=8)
    ax.text(-20, y - 0.2, label, ha="right", va="center", color=INK, fontsize=10)


def main():
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axes[0]
    draw_row(ax, 2.6, windows("aligned"), BLUE, "贴齐文末")
    draw_row(ax, 1.5, windows("short"), AQUA, "短尾窗")
    ws = windows("aligned")
    for (s0, e0), (s1, e1) in zip(ws[:-1], ws[1:]):
        ax.add_patch(Rectangle((s1, 0.9), e0 - s1, 2.4, fc="#eda100", alpha=0.13, ec="none"))
    ax.set_xlim(-130, N + 20)
    ax.set_ylim(0.8, 3.3)
    ax.set_yticks([])
    ax.set_xlabel("字符位置（文章长 900 字，窗口 400，重叠 100，步长 300）", color=INK2)
    ax.set_title("(a) 窗口规划：末窗两种规则，淡黄为相邻窗口的重叠区", loc="left", fontsize=10.5, color=INK)
    ax.spines["left"].set_visible(False)

    ax = axes[1]
    ax.add_patch(Rectangle((0, 2.5), 400, 0.4, fc=BLUE, ec="white", lw=1.5, alpha=0.85))
    ax.add_patch(Rectangle((300, 2.0), 400, 0.4, fc=BLUE, ec="white", lw=1.5, alpha=0.85))
    ax.text(-20, 2.5, "第一遍", ha="right", va="center", color=INK)
    for x in (290, 310, 395, 405):
        ax.plot([x], [2.2], marker="v", color=ORANGE, ms=6)
    ax.annotate("接缝附近合并概率\n低于 0.9 的位置", (400, 2.15), (560, 1.45), color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.add_patch(Rectangle((200, 0.7), 400, 0.4, fc=ORANGE, ec="white", lw=1.5, alpha=0.85))
    ax.add_patch(Rectangle((328, 0.62), 144, 0.56, fc="none", ec=INK, lw=1.0, ls="--"))
    ax.text(-20, 0.9, "第二遍补窗", ha="right", va="center", color=INK)
    ax.text(400, 0.38, "只重标距补窗两端各 128 字以外的位置", ha="center", va="top", color=INK2, fontsize=8.5)
    ax.set_xlim(-130, N + 20)
    ax.set_ylim(0.0, 3.1)
    ax.set_yticks([])
    ax.set_xlabel("字符位置", color=INK2)
    ax.set_title("(b) 第二遍接缝补窗：只对第一遍没有把握的接缝位置重新切窗", loc="left", fontsize=10.5, color=INK)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig1_method")


if __name__ == "__main__":
    main()
