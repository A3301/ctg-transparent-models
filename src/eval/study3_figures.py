"""Figures for Plans B-E (docs/analiz_plani_v3.md §6.2, v3_ek1, v3_ek2), journal artwork style (src/eval/figstyle.py).

- fig5_event_f1_methods: event F1 of the concept sources (A: CTU-UHB event F1, B: FHRMA difference from the extractor)
- fig4_robustness:       AUROC of the outcome models across the robustness settings
- fig3_shape_functions:  shape functions of the additive model
(The calibration / decision-curve figure is drawn in src/eval/study3_extra.py; src/eval/manuscript_figures.py
redraws the figures of the manuscript.)

Run: .venv\\Scripts\\python -m src.eval.study3_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import figstyle as FS  # noqa: E402

FS.setup()
OUT = ROOT / "figures" / "study3"
NAMES = {"sd2": "SD2, Poincaré (bpm)", "fhr_missing": "FHR signal loss (fraction)",
         "n_dec_other": "Unclassified decel. (n)", "n_dec_variable": "Variable decel. (n)",
         "n_dec_prolonged": "Prolonged decel. (n)", "uc_longest_gap_s": "Longest UC gap (s)",
         "reduced_var_min": "Reduced variability (min)", "lf": "LF power (log)",
         "vlf": "VLF power, 0–0.03 Hz (log)", "baseline": "Baseline (bpm)", "stv_ms": "STV (ms)",
         "n_acc": "Accelerations (n)", "fhr_longest_gap_s": "Longest FHR gap (s)", "dec_area": "Deceleration area",
         "n_dec": "Decelerations (n)",
         "dec_max_depth": "Deepest decel. (bpm)", "late_frac_contr": "Late decel. per contraction"}


def save(fig, name):
    FS.save(fig, OUT, name)


# Default appearance of the shared figures; src/eval/manuscript_figures.py passes its own values (style=...).
SHAPE_STYLE = dict(fold_color=FS.MUTED, fold_lw=0.6, med_color=FS.BLUE, med_lw=1.6, zero_color=FS.INK2, zero_lw=0.6,
                   title_size=7.5, tick_size=7, legend_size=7.5, grid_lw=0.6, height=3.55)
EVENT_STYLE = dict(ci_lw=1.3, ms=5, mew=0.6, grid_lw=0.6, ref_color=FS.REF, ref_lw=1.0, note_size=7, note_color=FS.INK2,
                   legend_size=7.2, legend_x=0.55)


def _grid(ax, axis, lw):
    ax.grid(axis=axis, color=FS.GRID, lw=lw)
    ax.set_axisbelow(True)


def _rows(ax, n):
    """Light alternating bands behind grouped rows (inverted y axis)."""
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0, lw=0)


def shape_figure(style=None):
    """Shape functions of the 8 variables with the largest contribution (10 outer folds of repeat 0, median).
    Count variables get integer x ticks."""
    st = {**SHAPE_STYLE, **(style or {})}
    R = json.loads((ROOT / "results/study3/study3_results.json").read_text(encoding="utf-8"))
    curves = {c: cs for c, cs in np.load(ROOT / "results/study3/shape_curves_r0.npy", allow_pickle=True)}
    top = R["shape_functions"]["top8"]
    fig, axes = plt.subplots(2, 4, figsize=(FS.WIDTH, st["height"]), sharey=True)
    for ax, c in zip(axes.ravel(), top):
        for g, v in curves[c]:
            ax.plot(g, v, color=st["fold_color"], lw=st["fold_lw"], alpha=0.9, zorder=2)
        g0 = np.array(curves[c][0][0])
        med = np.median(np.array([np.interp(g0, np.array(g), np.array(v)) for g, v in curves[c]]), 0)
        ax.plot(g0, med, color=st["med_color"], lw=st["med_lw"], zorder=3)
        ax.axhline(0, color=st["zero_color"], lw=st["zero_lw"], ls=(0, (3, 2)), zorder=1)
        ax.set_title(NAMES.get(c, c), fontsize=st["title_size"], color=FS.INK, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=NAMES.get(c, c).endswith("(n)")))
        ax.tick_params(labelsize=st["tick_size"])
        _grid(ax, "y", st["grid_lw"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ log-odds of pH < 7.05")
    axes[0, 0].set_ylim(-2.2, 2.4)
    leg = [Line2D([], [], color=st["fold_color"], lw=max(st["fold_lw"], 0.8), label="Each of the 10 outer folds (first repeat)"),
           Line2D([], [], color=st["med_color"], lw=st["med_lw"], label="Median across folds")]
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0, h_pad=0.8)
    fig.legend(handles=leg, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=st["legend_size"])
    save(fig, "fig3_shape_functions")


def robustness_figure():
    """Plan D: AUROC of the transparent models vs the CNN ensemble across the 8 settings."""
    D = json.loads((ROOT / "results/study3/planD_results.json").read_text(encoding="utf-8"))
    order = ["W60", "W30", "W90", "S1_60", "all552", "ph715", "bdecf12", "apgar5"]
    lab = {"W60": "Primary: last 60 min", "W30": "Last 30 min", "W90": "Last 90 min", "S1_60": "First stage, last 60 min",
           "all552": "All 552 recordings", "ph715": "Outcome pH ≤ 7.15", "bdecf12": "Outcome BDecf ≥ 12 mmol/L",
           "apgar5": "Outcome 5-min Apgar < 7"}
    fig, ax = plt.subplots(figsize=(FS.WIDTH, 3.5))
    y = np.arange(len(order))
    _rows(ax, len(order))
    ax.axvline(0.5, color=FS.INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(0.505, -0.62, "chance", fontsize=7, color=FS.INK2, va="center")
    handles = []
    for k, m in enumerate(("cnn", "transp_lr", "transp_gam")):
        name, col, mk = FS.MODEL[m]
        v = np.array([D[s][m]["auroc"] for s in order])
        yy = y + (k - 1) * 0.24
        ax.hlines(yy, v[:, 1], v[:, 2], color=col, lw=1.3, zorder=2)
        h, = ax.plot(v[:, 0], yy, mk, color=col, ms=5, mec="white", mew=0.6, zorder=3, label=name)
        handles.append(h)
    ax.set_yticks(y, [f"{lab[s]} ({D[s]['n_pos']} events)" for s in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.5, -0.9)
    ax.set_xlim(0.2, 0.95)
    ax.set_xlabel("AUROC (repeat-averaged out-of-fold predictions, 95% bootstrap CI)")
    FS.value_grid(ax, "x")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0))
    save(fig, "fig4_robustness")


def event_f1_figure(style=None):
    """Event-level F1 figure. A: event F1 vs CTU-UHB expert annotations (95% CI) for the four concept sources;
    B: difference in event F1 from the rule-based extractor on the external FHRMA consensus set (95% CI)."""
    st = {**EVENT_STYLE, **(style or {})}
    S2 = json.loads((ROOT / "results/study2/study2_results.json").read_text(encoding="utf-8"))["events"]
    S3 = json.loads((ROOT / "results/study3/study3_results.json").read_text(encoding="utf-8"))
    SE = json.loads((ROOT / "results/study3/planE_results.json").read_text(encoding="utf-8"))
    F2 = json.loads((ROOT / "results/study2/study2_fhrma.json").read_text(encoding="utf-8"))
    # CTU-UHB: fold-specific thresholds (deviation log #15), results/study3/foldwise_events.json
    FW = json.loads((ROOT / "results/study3/foldwise_events.json").read_text(encoding="utf-8"))["events"]
    ctu = [("nesy2", FW, "nesy2"), ("det", FW, "det"), ("detk", FW, "detk"), ("rule", FW, "rule")]
    fh = {"nesy2": {e: (F2[e]["diff_f1"], *F2[e]["ci"]) for e in ("ACC", "DEC")},
          "det": {e: (S3["detector_fhrma"][e]["diff"], S3["detector_fhrma"][e]["lo"], S3["detector_fhrma"][e]["hi"])
                  for e in ("ACC", "DEC")},
          "detk": {e: (SE["E2_fhrma"][e]["diff"], SE["E2_fhrma"][e]["lo"], SE["E2_fhrma"][e]["hi"]) for e in ("ACC", "DEC")}}

    fig = plt.figure(figsize=(FS.WIDTH, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.5, left=0.14, right=0.99, top=0.83, bottom=0.17)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    ev = ["ACC", "DEC", "UC"]
    y = np.arange(len(ev))
    _rows(axA, len(ev))
    handles = []
    for k, (key, src, field) in enumerate(ctu):
        name, col, mk = FS.SOURCE[key]
        v = np.array([src[f"{field}|{e}"]["f1_ci"] for e in ev])
        yy = y + (k - 1.5) * 0.19
        axA.hlines(yy, v[:, 1], v[:, 2], color=col, lw=st["ci_lw"], zorder=2)
        h, = axA.plot(v[:, 0], yy, mk, color=col, ms=st["ms"], mec="white", mew=st["mew"], zorder=3, label=name)
        handles.append(h)
    axA.set_yticks(y, ["Accelerations", "Decelerations", "Contractions"])
    axA.tick_params(axis="y", length=0)
    axA.set_ylim(len(ev) - 0.5, -0.5)
    axA.set_xlim(0.2, 0.9)
    axA.set_xlabel("Event-level F1, fold-specific thresholds (95% CI)")
    _grid(axA, "x", st["grid_lw"])
    FS.panel_title(axA, "A", "CTU-UHB expert annotations")

    evb = ["ACC", "DEC"]
    yb = np.arange(len(evb))
    _rows(axB, len(evb))
    axB.axvline(0, color=st["ref_color"], lw=st["ref_lw"], zorder=1)
    axB.text(-0.006, -0.47, "Extractor", fontsize=st["note_size"], color=st["note_color"], ha="right", va="top")
    for k, key in enumerate(("nesy2", "det", "detk")):
        name, col, mk = FS.SOURCE[key]
        v = np.array([fh[key][e] for e in evb])
        yy = yb + (k - 1) * 0.22
        axB.hlines(yy, v[:, 1], v[:, 2], color=col, lw=st["ci_lw"], zorder=2)
        axB.plot(v[:, 0], yy, mk, color=col, ms=st["ms"], mec="white", mew=st["mew"], zorder=3)
    axB.set_yticks(yb, ["Accelerations", "Decelerations"])
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(len(evb) - 0.5, -0.5)
    axB.set_xlim(-0.42, 0.06)
    axB.set_xlabel("F1 difference from the extractor (95% CI)")
    _grid(axB, "x", st["grid_lw"])
    FS.panel_title(axB, "B", "FHRMA consensus (external)")

    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(st["legend_x"], 1.01), fontsize=st["legend_size"])
    save(fig, "fig5_event_f1_methods")


def main():
    shape_figure()
    robustness_figure()
    event_f1_figure()


if __name__ == "__main__":
    main()
