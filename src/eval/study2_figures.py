"""Figures of the end-to-end model analysis (docs/analiz_plani_v2.md).

Reads only the aggregate result files written by study2_eval / study2_fhrma, plus record-level
quantities recomputed in memory for Fig. 2B-C (not written to disk: Romagnoli-derived, CC BY-NC-ND).
Run: .venv\\Scripts\\python -m src.eval.study2_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.data.inputs import labels  # noqa: E402

from src.eval import figstyle as FS  # noqa: E402

OUT = ROOT / "figures" / "study2"
FS.setup()
C1, C2 = FS.BLUE, FS.ORANGE            # kept for modules that import these names
INK, INK2, GRID, SURF = FS.INK, FS.INK2, FS.GRID, "white"


def save(fig, name):
    FS.save(fig, OUT, name)


def fig1(R, F):
    """Event-level F1 with 95% CI: CTU-UHB (quality region) and FHRMA (external)."""
    rows = []
    for ev in ("ACC", "DEC", "UC"):
        for m in ("nesy2", "rule"):
            f1, lo, hi = R["events"][f"{m}|{ev}"]["f1_ci"]
            rows.append(("CTU-UHB", ev, m, f1, lo, hi))
    # FHRMA: point estimates only for each method (CI was computed for the difference)
    for ev in ("ACC", "DEC"):
        for m in ("nesy2", "rule"):
            rows.append(("FHRMA (external)", ev, m, F[ev][m]["f1"], np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["set", "event", "model", "f1", "lo", "hi"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True, gridspec_kw={"width_ratios": [3, 2]})
    for ax, (s, g) in zip(axes, df.groupby("set", sort=False)):
        ev = list(dict.fromkeys(g.event))
        for k, (m, col, mk, lab) in enumerate((("nesy2", C1, "o", "NeSy-E2E (expert-supervised)"),
                                                ("rule", C2, "s", "Rule-based FIGO extractor"))):
            gg = g[g.model == m].set_index("event").loc[ev]
            y = np.arange(len(ev)) + (0.15 if k == 0 else -0.15)
            if gg.lo.notna().all():
                ax.hlines(y, gg.lo, gg.hi, color=col, lw=2)
            ax.plot(gg.f1, y, mk, color=col, ms=7, mec=SURF, mew=1.5, label=lab)
        ax.set_yticks(np.arange(len(ev)), ev)
        ax.set_title(s, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Event-level F1")
        ax.invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig1_event_f1")


# Default appearance of fig2; src/eval/manuscript_figures.py passes its own values (style=...).
FIG2_STYLE = dict(e2e=FS.ORANGE, comp=FS.REF, pos=FS.BLUE, neg=FS.MUTED, ms=5.5, ms_comp=5.2, mew=0.6, mew_comp=1.1,
                  pt_ms=3.4, bg_ms=2.4, med_lw=1.2, med_ms=3.6, iqr_lw=1.0, ref_color=INK2, ref_lw=0.8, grid_lw=0.6,
                  note_size=7, note_color=INK2, legend_size=7, height=2.75,
                  grid=dict(width_ratios=[1.2, 1.0, 1.1], wspace=0.55, left=0.19, right=0.975))   # fits 6.25 in


def fig2(R, *, comparator_label="Weak-label NeSy", output=None, style=None):
    """Concept/reference figure. A: window-concept AUROC vs outcome AUROC; B: expert event density vs FHR loss;
    C: share of the window masked from expert supervision, by outcome."""
    st = {**FIG2_STYLE, **(style or {})}
    W = R["window_concepts"]
    cats = [("accel", "Acceleration"), ("early", "Early deceleration"), ("rep_any", "Recurrent deceleration")]
    v2 = [W[k]["auroc_nesy2"] for k, _ in cats] + [R["H3"]["auroc_nesy2"][0]]
    v1 = [W[k]["auroc_nesy1"] for k, _ in cats] + [json.loads((ROOT / "results/cv/nesy_ens_W60/summary.json")
                                                             .read_text())["auroc"][0]]
    names = [f"{n}\n(concept)" for _, n in cats] + ["pH < 7.05\n(outcome)"]

    T = load_targets()
    recs = [str(r) for r in T["records"]]
    wq = pd.read_csv(ROOT / "results/preprocess/window_quality.csv", dtype={"record": str})
    wq = wq[wq.window == "W60"].set_index("record")
    lab = labels("y_ph705")
    study = set()                                     # the 536 analysed records (as in study2_eval)
    for fp in (ROOT / "results/cv/nesy2_W60").glob("r0_f*.npz"):
        study |= {str(x) for x in np.load(fp, allow_pickle=True)["records_te"]}
    assert len(study) == 536
    rows = []
    for i, r in enumerate(recs):
        if not T["annotated"][i] or r not in study:
            continue
        cov_h = T["cov"][i].sum() * 4 / 3600
        n_ev = int(((T["ev_rec"] == i) & (T["ev_label"] <= 1)).sum())
        rows.append(dict(density=n_ev / max(cov_h, 1e-6), miss=float(wq.at[r, "fhr_missing"]), y=int(lab[r]),
                         masked=1 - float(T["q"][i].sum() / max(T["cov"][i].sum(), 1))))
    dd = pd.DataFrame(rows)

    fig = plt.figure(figsize=(FS.WIDTH, st["height"]))
    gs = fig.add_gridspec(1, 3, top=0.80, bottom=0.2, **st["grid"])
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))

    def grid(ax, axis):
        ax.grid(axis=axis, color=GRID, lw=st["grid_lw"])
        ax.set_axisbelow(True)

    # A: concept fidelity vs outcome discrimination
    y = np.arange(len(names))
    axA.axhspan(2.5, 3.5, color="#f3f3f3", zorder=0, lw=0)
    axA.axvline(0.5, color=st["ref_color"], lw=st["ref_lw"], ls=(0, (3, 2)), zorder=1)
    axA.text(0.505, -0.62, "chance", fontsize=st["note_size"], color=st["note_color"], va="center")
    hE, = axA.plot(v2, y - 0.15, "s", color=st["e2e"], ms=st["ms"], mec="white", mew=st["mew"], zorder=3,
                   label="NeSy-E2E (end-to-end)")
    hS, = axA.plot(v1, y + 0.15, "o", mfc="white", mec=st["comp"], mew=st["mew_comp"], ms=st["ms_comp"], zorder=3,
                   label=comparator_label)
    axA.set_yticks(y, names)
    axA.set_ylim(3.6, -0.9)
    axA.set_xlim(0.4, 0.85)
    axA.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axA.set_xlabel("AUROC")
    grid(axA, "x")
    FS.panel_title(axA, "A", "Concepts vs outcome")

    # B: reference density vs signal loss (all annotated records; binned median and IQR)
    g0, g1 = dd[dd.y == 0], dd[dd.y == 1]
    axB.plot(g0.miss * 100, g0.density, "o", color=st["neg"], ms=st["bg_ms"], alpha=0.55, mec="none", zorder=2)
    hA, = axB.plot(g1.miss * 100, g1.density, "s", color=st["pos"], ms=st["pt_ms"], mec="white", mew=0.4, zorder=3,
                   label="pH < 7.05")
    edges = np.array([0, 5, 10, 20, 30, 100])
    mids, med, q1, q3 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = dd[(dd.miss * 100 >= lo) & (dd.miss * 100 < hi)]
        if len(g) >= 5:
            mids.append(g.miss.median() * 100)
            med.append(g.density.median())
            q1.append(g.density.quantile(0.25))
            q3.append(g.density.quantile(0.75))
    mids, med, q1, q3 = map(np.array, (mids, med, q1, q3))
    axB.vlines(mids, q1, q3, color=INK, lw=st["iqr_lw"], zorder=4)
    hM, = axB.plot(mids, med, "-o", color=INK, lw=st["med_lw"], ms=st["med_ms"], mfc="white", mec=INK, mew=1.0, zorder=5,
                   label="Median (IQR)")
    axB.set_xlabel("FHR signal loss in window (%)")
    axB.set_ylabel("Expert ACC + DEC per hour")
    axB.set_xlim(-2, 52)
    axB.set_ylim(-1, 37)
    axB.set_yticks([0, 10, 20, 30])
    grid(axB, "y")
    FS.panel_title(axB, "B", "Reference density")

    # C: share of the window masked from expert supervision
    # masked share is bimodal: below 50 % (partial gaps) or 100 % (record fails the record-level quality criterion)
    assert not ((dd.masked >= 0.5) & (dd.masked < 0.9999)).any()
    bins = [(0, 0.25, "<25"), (0.25, 0.5, "25–49"), (0.9999, 1.01, "100")]
    share = {}
    for yy, g in ((0, g0), (1, g1)):
        share[yy] = [100 * np.mean((g.masked >= lo) & (g.masked < hi)) for lo, hi, _ in bins]
    x = np.arange(len(bins))
    w = 0.38
    hN = axC.bar(x - w / 2 - 0.01, share[0], w, color=st["neg"], label="pH ≥ 7.05", zorder=2)
    axC.bar(x + w / 2 + 0.01, share[1], w, color=st["pos"], label="pH < 7.05", zorder=2)
    axC.set_xticks(x, [b[2] for b in bins])
    axC.set_xlabel("Share of window masked (%)")
    axC.set_ylabel("Recordings (%)")
    axC.set_ylim(0, 75)
    m0, m1 = g0.masked.mean() * 100, g1.masked.mean() * 100
    # two lines inside the axes (a single line overflows the right edge of the figure)
    axC.text(0.02, 0.99, f"Mean: {m1:.0f}% (pH < 7.05)\nvs {m0:.0f}% (pH ≥ 7.05)", transform=axC.transAxes,
             fontsize=st["note_size"], va="top", color=INK, linespacing=1.2)
    grid(axC, "y")
    FS.panel_title(axC, "C", "Masked windows")

    fig.legend([hE, hS, hN, hA, hM], ["NeSy-E2E (end-to-end)", comparator_label, "pH ≥ 7.05", "pH < 7.05",
                                     "Median (IQR)"],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=5, fontsize=st["legend_size"], handlelength=1.2)
    if output is None:
        save(fig, "fig2_dissociation_reference")
    else:
        FS.save(fig, output.parent, output.name)
    return dict(n=len(dd), n_pos=int(dd.y.sum()), masked_mean=dd.groupby("y").masked.mean().round(3).to_dict(),
                masked_median=dd.groupby("y").masked.median().round(3).to_dict(),
                masked_share_bins={k: [round(v, 1) for v in s] for k, s in share.items()})


def main():
    R = json.loads((ROOT / "results/study2/study2_results.json").read_text(encoding="utf-8"))
    F = json.loads((ROOT / "results/study2/study2_fhrma.json").read_text(encoding="utf-8"))
    fig1(R, F)
    print(fig2(R))


if __name__ == "__main__":
    main()
