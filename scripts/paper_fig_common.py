"""Shared style for the paper figures: Chinese font, fixed colours, save helper.

Colours follow the dataviz reference palette (slots 1 to 3 plus neutral grays); text uses ink
tokens, never a series colour. Figures are written to figures/paper/ as PNG and PDF.
"""
from __future__ import annotations

import os

import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.edgecolor"] = "#52514e"
mpl.rcParams["axes.labelcolor"] = "#0b0b0b"
mpl.rcParams["xtick.color"] = "#52514e"
mpl.rcParams["ytick.color"] = "#52514e"
mpl.rcParams["grid.color"] = "#e3e2de"
mpl.rcParams["grid.linewidth"] = 0.6
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e3e2de"
OUT = os.path.join("figures", "paper")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    print("saved", os.path.join(OUT, name))
