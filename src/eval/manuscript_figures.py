"""Figures of the manuscript "Expert-referenced interpretability in intrapartum cardiotocography models for fetal acidaemia".

- fig1_design: study design (two routes to interpretability, one black box, shared evaluation);
- fig2_forest: acidaemia AUROC (95% CI) of all nine models, grouped by route; values read from the result files;
- fig3_concept_agreement: window concepts vs outcome, expert reference density and masked windows;
- fig4_event_f1: event-level F1 of the four concept sources (CTU-UHB and FHRMA);
- fig5_shape_functions: shape functions of the additive model;
- fig6_calibration_dca: calibration and decision curves with route colours (transparent models in blue shades,
  the CNN ensemble in grey), so that orange always means the concept route in this manuscript.

Colours by route: concept route = orange, transparent variables = blue (two shades), black box = grey.
Every series also has its own marker, so identity never depends on colour alone.

Run: python -m src.eval.manuscript_figures   (with the project .venv)
Output: figures/manuscript/fig1-fig6 (*.png, *.pdf)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import figstyle as FS  # noqa: E402

OUT = ROOT / "figures/manuscript"
# Manuscript palette: deeper versions of the route colours, so that thin lines and small markers stand out by
# contrast (concept route = orange family, transparent route = blue family, black box = dark grey).
CONCEPT, CONCEPT_LIGHT, CONCEPT_DARK = "#d0501a", "#ec8a26", "#8f3210"
TRANSP, TRANSP2, BLACK = "#1b5fbf", "#0c3570", "#3a3a3a"
TINT = {CONCEPT: "#fbdcc9", TRANSP: "#d6e6f8", BLACK: "#e3e3e3"}     # lane fills of Fig. 1
# Shared line, marker and type sizes (pt) of Figs 1-6; figures are drawn at the 6.25-in text width (100 %).
LW, CI_LW, MS, MEW, GRID_LW, REF_LW = 1.1, 1.0, 4.5, 0.4, 0.4, 0.7
NOTE, LEG = 7.5, 7.5                     # in-plot annotations, legends
R = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731


def _grid(ax, axis):
    ax.grid(axis=axis, color=FS.GRID, lw=GRID_LW)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------------------- study design
# Fig. 1 is drawn in point coordinates (1 data unit = 1 pt at the printed size).
def _box(ax, x, y, w, h, text, edge, face="white", header=False, size=7.0, lw=0.9):
    """Rounded box; with header=True the first line is a bold heading above the body lines."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", lw=lw, ec=edge, fc=face, zorder=2,
                                clip_on=False))
    cx, cy = x + w / 2, y + h / 2
    if not header:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=FS.INK, zorder=3, linespacing=1.15)
        return
    head, *body = text.split("\n")
    hs, gap = size + 0.4, 2.5
    hh, bh = 1.08 * hs, 1.086 * size + (len(body) - 1) * 1.079 * size     # text heights (pt), linespacing 1.15
    top = cy + (hh + gap + bh) / 2
    ax.text(cx, top, head, ha="center", va="top", fontsize=hs, fontweight="bold", color=FS.INK, zorder=3)
    ax.text(cx, top - hh - gap, "\n".join(body), ha="center", va="top", fontsize=size, color=FS.INK, zorder=3,
            linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color=FS.INK2, lw=0.9, ls="-", ms=7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=ms, shrinkA=0, shrinkB=0))


def design():
    W, H = 445.0, 268.0                                   # pt; with the save padding the file is 6.25 in wide
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    BODY, LANE, COL = 7.0, 7.6, 8.2                     # box text, lane headings, column headings (pt)
    BUS, FRAME = "#8c8c8c", FS.INK2

    # columns (x in pt): data | distribution bus | three boxes per route | collection bus | evaluation
    dx, dw, bus_in = 4.0, 70.0, 84.0
    x0, widths, gap, h = 104.0, (80.0, 84.0, 54.0), 16.0, 38.0
    x_end = x0 + sum(widths) + 2 * gap
    bus_out, ex, ew = x_end + 7.0, x_end + 15.0, 75.0
    col_y = 262.0
    ax.text(dx + dw / 2, col_y, "Data and references", ha="center", va="center", fontsize=COL, fontweight="bold", color=FS.INK)
    ax.text((x0 + x_end) / 2, col_y, "Routes to an acidaemia risk", ha="center", va="center", fontsize=COL,
            fontweight="bold", color=FS.INK)
    ax.text(ex + ew / 2, col_y, "Evaluation", ha="center", va="center", fontsize=COL, fontweight="bold", color=FS.INK)

    # route lanes: (bottom y, heading, colour, box texts)
    lanes = [
        (200.0, "Concept route: end-to-end", CONCEPT,
         ("Concept heads:\nbaseline, variability,\n10 event concepts", "FIGO 2015 logic,\nnoisy-OR (Eq. 1)", "NeSy-E2E\nrisk")),
        (142.0, "Concept route: separate detectors", CONCEPT,
         ("Event detectors:\nsequential or\nrule-guided", "23 FIGO features from\ndetected events\n+ HRV, signal quality",
          "Additive\nrisk")),
        (84.0, "Transparent route", TRANSP,
         ("Deterministic code:\nFIGO extractor,\nHRV, signal quality", "30 variables\n(23 + 3 + 4)", "Additive or\nlinear (Eq. 2)")),
        (26.0, "Black box", BLACK, ("1-D residual CNN on\nFHR, UC and masks", "5-member\nensemble", "CNN\nrisk")),
    ]
    centres = []
    for y, head, col, texts in lanes:
        ax.text(x0, y + h + 7.5, head, ha="left", va="center", fontsize=LANE, fontweight="bold", color=col)
        x = x0
        for i, (text, w) in enumerate(zip(texts, widths)):
            _box(ax, x, y, w, h, text, col, face=TINT[col], size=BODY)
            if i:
                _arrow(ax, x - gap + 0.8, y + h / 2, x - 0.6, y + h / 2, color=col)
            x += w + gap
        centres.append(y + h / 2)

    # data column: expert events (top, supervise the concept route), CTU-UHB (feeds every route), FHRMA (external)
    _box(ax, dx, 190.0, dw, 48.0, "Expert events\nCTU-UHB\nannotations,\n513 recordings", FRAME, header=True, size=BODY)
    _box(ax, dx, 104.0, dw, 56.0, "CTU-UHB\n536 recordings\n38 with pH < 7.05\nlast 60 min\nFHR + UC", FRAME,
         header=True, size=BODY)
    _box(ax, dx, 26.0, dw, 48.0, "FHRMA\nconsensus,\n66 recordings\n(external)", FRAME, header=True, size=BODY)

    # CTU-UHB -> routes: distribution bus
    ax.plot([bus_in, bus_in], [centres[-1], centres[0]], color=BUS, lw=0.8, zorder=1, solid_capstyle="butt")
    ax.plot([dx + dw, bus_in], [132.0, 132.0], color=BUS, lw=0.8, zorder=1, solid_capstyle="butt")
    for c in centres:
        _arrow(ax, bus_in, c, x0 - 0.6, c, color=BUS, lw=0.8)
    # expert events supervise the two concept routes (dashed orange)
    _arrow(ax, dx + dw, 207.0, x0 - 0.6, 207.0, color=CONCEPT, lw=0.9, ls=(0, (3, 2)), ms=6.5)
    _arrow(ax, dx + dw, 197.0, x0 - 0.6, 172.0, color=CONCEPT, lw=0.9, ls=(0, (3, 2)), ms=6.5)
    ax.text(bus_in - 4.0, 175.0, "supervision", ha="right", va="center", fontsize=BODY, fontstyle="italic", color=CONCEPT)

    # routes -> evaluation: collection bus
    ax.plot([bus_out, bus_out], [centres[-1], centres[0]], color=BUS, lw=0.8, zorder=1, solid_capstyle="butt")
    for c in centres:
        ax.plot([x_end - 4.0, bus_out], [c, c], color=BUS, lw=0.8, zorder=1, solid_capstyle="butt")
    evals = [(190.0, "Concept fidelity\nevent F1 (Eq. 4)\nagainst CTU-UHB\nexperts and FHRMA"),
             (108.0, "Risk dependence\nexpert concept\nsubstitution (Eq. 6)"),
             (26.0, "Outcome\nAUROC,\ncalibration (Eq. 3),\nnet benefit (Eq. 5)")]
    for y, text in evals:
        _box(ax, ex, y, ew, 48.0, text, FRAME, header=True, size=BODY)
        _arrow(ax, bus_out, y + 24.0, ex - 0.6, y + 24.0, color=BUS, lw=0.8)
    ax.text(W / 2, 9.0, "5 × 10 record-level nested cross-validation  ·  paired record bootstrap (2,000 resamples)\n"
            "thresholds and calibration from training-side data only", ha="center", va="center", fontsize=7.0,
            color=FS.INK, linespacing=1.25)
    FS.save(fig, OUT, "fig1_design")


# ---------------------------------------------------------------------------------------------- forest plot
def forest():
    S2, S3, E, X = (R("results/study2/study2_results.json"), R("results/study3/study3_results.json"),
                    R("results/study3/planE_results.json"), R("results/study3/extra_results.json"))
    D = S3["discrimination"]
    rows = [  # (group, label, [est, lo, hi], colour, marker)
        ("Concept route", "NeSy-E2E (end-to-end)", S2["H3"]["auroc_nesy2"], CONCEPT, "s"),
        ("Concept route", "Additive, sequential-detector concepts", D["transpdet_gam"]["auroc"], CONCEPT, "o"),
        ("Concept route", "Additive, rule-guided-detector concepts", E["E3_transpdetk_gam"]["auroc"], CONCEPT, "^"),
        ("Transparent route", "Additive model, 30 variables (primary)", D["transp_gam"]["auroc"], TRANSP2, "s"),
        ("Transparent route", "Linear model, 30 variables", D["transp_lr"]["auroc"], TRANSP, "o"),
        ("Transparent route", "HRV linear model, 3 variables (post hoc)", X["hrv_lr"]["auroc_ci"], TRANSP, "^"),
        ("Transparent route", "HRV support-vector model, 3 variables", D["georgoulas"]["auroc"], TRANSP, "v"),
        ("Transparent route", "FIGO-feature model, 23 variables", D["figo_lr"]["auroc"], TRANSP, "D"),
        ("Black box", "CNN ensemble", D["cnn_bayes"]["auroc"], BLACK, "D"),
    ]
    gcol = {"Concept route": CONCEPT, "Transparent route": TRANSP2, "Black box": BLACK}
    y, ys, heads, group = 0.0, [], [], None
    for g, *_ in rows:
        if g != group:
            y -= 0.5 if group is not None else 0.0
            heads.append((g, y))
            y -= 0.95
            group = g
        ys.append(y)
        y -= 1.0
    top, bottom = heads[0][1] + 0.8, ys[-1] - 0.7
    FS.setup()
    fig = plt.figure(figsize=(FS.WIDTH, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.25, 2.95, 1.1], wspace=0.02, left=0.006, right=0.99, top=0.97, bottom=0.14)
    axl, ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for a in (axl, axr):
        a.set_xlim(0, 1)
        a.axis("off")
    for a in (axl, ax, axr):
        a.set_ylim(bottom, top)
    cnn = D["cnn_bayes"]["auroc"][0]
    for xx in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        ax.plot([xx, xx], [bottom, top - 0.9], color=FS.GRID, lw=GRID_LW, zorder=0)
    ax.plot([cnn, cnn], [bottom, top - 0.45], color=FS.INK2, lw=REF_LW, ls=(0, (3, 2)), zorder=1)
    ax.text(cnn, top - 0.35, "CNN ensemble", ha="center", va="bottom", fontsize=NOTE, color=FS.INK)
    for (g, label, v, col, mk), yy in zip(rows, ys):
        est, lo, hi = (float(z) for z in v[:3])
        ax.plot([lo, hi], [yy, yy], color=col, lw=CI_LW, solid_capstyle="butt", zorder=2)
        ax.plot(est, yy, mk, color=col, ms=MS + 0.3, mec="white", mew=MEW, zorder=3)
        axl.text(0.0, yy, label, ha="left", va="center", fontsize=7.5, color=FS.INK)
        axr.text(1.0, yy, f"{est:.3f} ({lo:.3f}–{hi:.3f})", ha="right", va="center", fontsize=7.5, color=FS.INK)
    for g, yy in heads:
        axl.text(0.0, yy, g, ha="left", va="center", fontsize=8, fontweight="bold", color=gcol[g])
    axr.text(1.0, heads[0][1], "AUROC (95% CI)", ha="right", va="center", fontsize=8, fontweight="bold", color=FS.INK)
    ax.set_xlim(0.62, 0.92)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_xticklabels(["0.65", "0.70", "0.75", "0.80", "0.85", "0.90"])
    ax.set_xlabel("Acidaemia AUROC (pH < 7.05; 38 events)")
    FS.save(fig, OUT, "fig2_forest")


# ---------------------------------------------------------------------------------------------- calibration / DCA
def calibration_dca():
    from src.eval.study2_eval import repeat_avg
    X = R("results/study3/extra_results.json")
    base = repeat_avg("cnn_ens_W60")
    recs = sorted(base.index)
    y = base.loc[recs].y.values.astype(int)
    P = {k: repeat_avg(v).loc[recs].p.values for k, v in {"cnn_bayes": "cnn_ens_W60", "transp_lr": "transp_lr_W60",
                                                          "transp_gam": "transp_gam_W60"}.items()}
    # (key, name, colour, marker, line style, markevery in panel B: staggered so the markers do not overlap)
    show = [("cnn_bayes", "CNN ensemble", BLACK, "D", "-", (2, 5)), ("transp_lr", "Linear model", TRANSP, "o", "-", (4, 5)),
            ("transp_gam", "Additive model", TRANSP2, "s", (0, (4, 1.5)), (0, 5))]
    th = np.array([float(t) for t in X["treat_all_net_benefit"]])
    FS.setup()
    fig, ax = plt.subplots(1, 2, figsize=(FS.WIDTH, 2.9), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.82)
    ax[0].plot([0, 0.4], [0, 0.4], color=FS.INK2, lw=REF_LW, ls=(0, (3, 2)), zorder=1)
    ax[0].text(0.33, 0.355, "ideal", rotation=45, fontsize=NOTE, color=FS.INK, ha="center", va="center")
    handles = []
    for k, name, col, mk, ls, every in show:
        o = np.argsort(P[k])
        bins = np.array_split(o, 10)
        h, = ax[0].plot([P[k][b].mean() for b in bins], [y[b].mean() for b in bins], marker=mk, ls=ls, color=col, lw=LW,
                        ms=MS, mec="white", mew=MEW, zorder=3, label=name)
        handles.append(h)
        ax[1].plot(th, [X[k]["net_benefit"][t] for t in X[k]["net_benefit"]], color=col, lw=LW, ls=ls, zorder=3,
                   marker=mk, ms=MS - 0.5, mec="white", mew=MEW, markevery=every)
    ax[0].set_xlim(0, 0.4)
    ax[0].set_ylim(-0.016, 0.4)          # small margin below 0 so that deciles with no event are not clipped
    ax[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax[0].set_xlabel("Predicted risk (decile mean)")
    ax[0].set_ylabel("Observed proportion, pH < 7.05")
    _grid(ax[0], "both")
    FS.panel_title(ax[0], "A", "Calibration")
    ax[1].axhline(0, color=FS.INK2, lw=REF_LW, zorder=1)
    ax[1].text(0.298, -0.0015, "treat none", fontsize=NOTE, color=FS.INK, ha="right", va="top")
    hT, = ax[1].plot(th, list(X["treat_all_net_benefit"].values()), color="#8c8c8c", lw=LW, ls=(0, (3, 2)), label="Treat all")
    ax[1].set_xlim(0, 0.305)
    ax[1].set_xticks([0, 0.1, 0.2, 0.3])
    ax[1].set_ylim(-0.01, 0.07)
    ax[1].set_xlabel("Threshold probability")
    ax[1].set_ylabel("Net benefit")
    _grid(ax[1], "y")
    FS.panel_title(ax[1], "B", "Decision curve")
    fig.legend(handles=handles + [hT], loc="upper center", ncol=4, bbox_to_anchor=(0.54, 1.0), fontsize=LEG)
    FS.save(fig, OUT, "fig6_calibration_dca")


def event_f1():
    """Event-level F1 figure (src.eval.study3_figures.event_f1_figure), redrawn with route colours: learned concept
    sources in orange shades, the rule-based extractor (the feature engine of the transparent route) in blue, also for
    its reference line in panel B."""
    from src.eval import study3_figures as S3F
    keep = dict(FS.SOURCE)
    FS.SOURCE.update({"nesy2": ("NeSy-E2E (end-to-end)", CONCEPT, "s"), "det": ("Sequential detector", CONCEPT_LIGHT, "o"),
                      "detk": ("Rule-guided detector", CONCEPT_DARK, "^"), "rule": ("Rule-based FIGO extractor", TRANSP, "D")})
    save0 = S3F.save
    S3F.save = lambda fig, name: FS.save(fig, OUT, "fig4_event_f1")
    try:
        S3F.event_f1_figure(style=dict(ci_lw=CI_LW, ms=MS, mew=MEW, grid_lw=GRID_LW, ref_color=TRANSP, ref_lw=LW,
                                       note_size=NOTE, note_color=FS.INK, legend_size=LEG, legend_x=0.53))
    finally:
        S3F.save = save0
        FS.SOURCE.clear()
        FS.SOURCE.update(keep)


def concept_agreement():
    """Figure 3: the concept/reference figure of src.eval.study2_figures, with the comparator named as in the text."""
    from src.eval import study2_figures as S2F
    S2F.fig2(R("results/study2/study2_results.json"), comparator_label="Weak-label NeSy",
             output=OUT / "fig3_concept_agreement",
             style=dict(e2e=CONCEPT, comp=BLACK, pos=TRANSP, neg=FS.MUTED, ms=MS, ms_comp=MS, mew=MEW, mew_comp=0.9,
                        pt_ms=3.4, bg_ms=2.4, med_lw=LW, med_ms=3.8, iqr_lw=CI_LW, ref_color=FS.INK2, ref_lw=REF_LW,
                        grid_lw=GRID_LW, note_size=NOTE, note_color=FS.INK, legend_size=LEG, height=2.8))


def shape_functions():
    """Figure 5: shape functions of the additive model (drawn by src.eval.study3_figures.shape_figure)."""
    from src.eval import study3_figures as S3F
    save0 = S3F.save
    S3F.save = lambda fig, name: FS.save(fig, OUT, "fig5_shape_functions")
    try:
        S3F.shape_figure(style=dict(fold_color="#9c9c9c", fold_lw=0.5, med_color=TRANSP, med_lw=1.2, zero_color=FS.INK2,
                                    zero_lw=REF_LW, title_size=7.5, tick_size=7.5, legend_size=LEG, grid_lw=GRID_LW,
                                    height=3.6))
    finally:
        S3F.save = save0


# (name, drawing function, inputs it needs when they are missing, e.g. in a public clone without the fold outputs)
FIGURES = (
    ("fig1_design", design, "nothing"),
    ("fig2_forest", forest, "the aggregate result files in results/study2 and results/study3"),
    ("fig4_event_f1", event_f1, "the aggregate result files in results/study2 and results/study3"),
    ("fig6_calibration_dca", calibration_dca, "fold prediction arrays from scripts/reproduce_paper.sh"),
    ("fig3_concept_agreement", concept_agreement,
     "the expert annotations and fold prediction arrays from scripts/reproduce_paper.sh"),
    ("fig5_shape_functions", shape_functions, "results/study3/shape_curves_r0.npy from scripts/reproduce_paper.sh"),
)


def main():
    FS.setup()
    for name, draw, needs in FIGURES:
        try:
            draw()
        except (FileNotFoundError, KeyError, OSError):
            plt.close("all")
            print(f"{name} skipped: needs {needs}")


if __name__ == "__main__":
    main()
