"""Shared style for journal artwork (Elsevier artwork rules).

- white background, Arial (falls back to DejaVu Sans), print-size type: 7.5-8.5 pt at final size;
- figures are drawn at the text width of the A4 manuscript (6.25 in = 159 mm), so Word embeds them at 100 %
  and the type keeps its size; the journal reproduces them at 90 / 140 / 190 mm without loss (vector PDF);
- vector PDF with embedded TrueType fonts (pdf.fonttype 42) and a 600-dpi PNG.

Colours: categorical slots 1-3 of the validated reference palette (blue, orange, aqua; colour-blind safe on a
white surface, worst CVD dE 9.2) plus a neutral grey for reference series. Every series also has its own marker
shape, so identity never depends on colour alone (grey-scale print).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF = "#4a4a4a"        # reference series (rule-based extractor, CNN ensemble)
MUTED = "#a6a6a6"      # de-emphasised raw data
INK, INK2, GRID = "#111111", "#555555", "#e6e6e6"
WIDTH = 6.25           # in; text width of the manuscript page (A4, 2.54 cm margins)

# one identity per concept source / model across all figures
SOURCE = {
    "nesy2": ("NeSy-E2E (end-to-end)", ORANGE, "s"),
    "det": ("Sequential detector", BLUE, "o"),
    "detk": ("Rule-guided detector", AQUA, "^"),
    "rule": ("Rule-based FIGO extractor", REF, "D"),
}
MODEL = {
    "cnn": ("CNN ensemble", REF, "D"),
    "transp_lr": ("Linear model", BLUE, "o"),
    "transp_gam": ("Additive model", ORANGE, "s"),
}


def setup() -> None:
    have_arial = any(f.name == "Arial" for f in font_manager.fontManager.ttflist)
    family = "Arial" if have_arial else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.bf": f"{family}:bold",
        "axes.edgecolor": INK2, "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
        "axes.titlepad": 5, "axes.labelpad": 3,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2,
        "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_title(ax, letter: str, title: str) -> None:
    """Bold panel letter followed by a short title, left-aligned over the plot area."""
    ax.set_title(rf"$\mathbf{{{letter}}}$   {title}", loc="left", fontsize=8.5, color=INK)


def value_grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
