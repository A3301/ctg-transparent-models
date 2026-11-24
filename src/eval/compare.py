"""Phase 5: main table, hypothesis tests (H1, H2), multiple-comparison family, figures.

usage: python -m src.eval.compare   (uses experiments listed in MAIN / FAMILY)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402
from src.eval.summarize import averaged, load_exp, summarize  # noqa: E402
from src.utils.repro import log_experiment  # noqa: E402

TAB = ROOT / "results" / "tables"
FIG = ROOT / "figures" / "results"
PROPOSED, H1_REF = "nesy_ens_W60", "cnn_ens_W60"
NAMES = {"nesy_ens_W60": "NeSy-Bayes (önerilen)", "cnn_ens_W60": "CNN-Bayes", "cnn_W60": "CNN",
         "gru_W60": "GRU", "transformer_W60": "Transformer", "mcnn_W60": "MCNN (Petrozziello 2019)",
         "fhrlinet_W60": "FHR-LINet (Mendis 2024)", "georgoulas_W60": "HRV-SVM (Georgoulas 2017)",
         "figo_lr_W60": "FIGO-LR", "figo_xgb_W60": "FIGO-XGB",
         "nesy_single_W60": "Abl: Bayesçi katman yok", "nesy_nologic_W60": "Abl: mantık katmanı yok",
         "nesy_nocsup_W60": "Abl: kavram denetimi yok", "nesy_rulesonly_W60": "Abl: yalnız kurallar"}
FAMILY = ["cnn_ens_W60", "figo_lr_W60", "figo_xgb_W60", "georgoulas_W60", "cnn_W60", "gru_W60",
          "transformer_W60", "mcnn_W60", "fhrlinet_W60"]
ABL = ["nesy_single_W60", "nesy_nologic_W60", "nesy_nocsup_W60", "nesy_rulesonly_W60"]


def fmt(v, d=3):
    return f"{v[0]:.{d}f} ({v[1]:.{d}f}–{v[2]:.{d}f})"


def get_summary(exp):
    p = ROOT / "results" / "cv" / exp / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    s = summarize(exp)
    p.write_text(json.dumps(s, indent=1))
    log_experiment("5", exp, json.loads((ROOT / "results" / "cv" / exp / "run_args.json").read_text()),
                   20261122, {k: s[k] for k in ("auroc", "auprc", "brier", "ece", "tpr_fpr10", "g_gmean")},
                   "summary of nested CV (repeat-averaged OOF)")
    return s


def available(exps):
    out = []
    for e in exps:
        d = ROOT / "results" / "cv" / e
        a = json.loads((d / "run_args.json").read_text()) if (d / "run_args.json").exists() else None
        if a is None:
            continue
        want = len(a["repeats"].split(",")) * len(a["folds"].split(","))
        if len(list(d.glob("r*_f*.npz"))) == want:
            out.append(e)
    return out


def main_table(exps):
    rows = []
    for e in exps:
        s = get_summary(e)
        rows.append({"Model": NAMES.get(e, e), "exp": e, "AUROC": fmt(s["auroc"]), "AUPRC": fmt(s["auprc"]),
                     "Se@Sp90": fmt(s["sens_at_spec90"]), "Se@Sp85": fmt(s["sens_at_spec85"]),
                     "TPR@FPR5/10/15/20": "/".join(f"{s[k][0]:.2f}" for k in ("tpr_fpr05", "tpr_fpr10", "tpr_fpr15", "tpr_fpr20")),
                     "G-mean (=QI)": fmt(s["g_gmean"]), "Se/Sp (G-mean eşiği)": f"{s['g_se'][0]:.2f}/{s['g_sp'][0]:.2f}",
                     "Brier": fmt(s["brier"]), "ECE": fmt(s["ece"]),
                     "Kal. eğimi": f"{s['cal_slope']:.2f}", "AUROC tekrar ort±SD": f"{np.mean(s['auroc_per_repeat']):.3f}±{np.std(s['auroc_per_repeat']):.3f}"})
    return pd.DataFrame(rows)


def preds(exp):
    a = averaged(load_exp(exp)).set_index("record")
    return a


def h1(prop=PROPOSED, ref=H1_REF, n=2000):
    a, b = preds(prop), preds(ref)
    idx = a.index.intersection(b.index)
    y, pa, pb = a.loc[idx, "y"].values, a.loc[idx, "p"].values, b.loc[idx, "p"].values
    res = {}
    res["dAUROC"] = M.paired_bootstrap_diff(y, pa, pb, M.auroc, n)
    res["dAUPRC"] = M.paired_bootstrap_diff(y, pa, pb, M.auprc, n)
    res["dBrier"] = M.paired_bootstrap_diff(y, pa, pb, M.brier, n)
    res["dECE"] = M.paired_bootstrap_diff(y, pa, pb, M.ece, n)
    res["delong"] = M.delong_test(y, pa, pb)
    lo = res["dAUROC"][1]
    res["H1a_noninferior_margin_-0.05"] = bool(lo > -0.05)
    res["H1a_superior"] = bool(lo > 0 and res["delong"][3] < 0.05)
    res["H1b_brier_better"] = bool(res["dBrier"][2] < 0)
    res["H1b_ece_better"] = bool(res["dECE"][2] < 0)
    return res


def family(prop=PROPOSED, refs=FAMILY):
    a = preds(prop)
    rows = []
    for r in refs:
        b = preds(r)
        idx = a.index.intersection(b.index)
        d, se, z, p = M.delong_test(a.loc[idx, "y"].values, a.loc[idx, "p"].values, b.loc[idx, "p"].values)
        rows.append({"Karşılaştırma": f"{NAMES[prop]} − {NAMES.get(r, r)}", "ΔAUROC": d, "SE": se, "z": z, "p (DeLong)": p})
    df = pd.DataFrame(rows)
    df["p (Holm)"] = M.holm(df["p (DeLong)"].values)
    return df


def figures(exps):
    FIG.mkdir(parents=True, exist_ok=True)
    P = {e: preds(e) for e in exps}
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, e in enumerate(exps):
        y, p = P[e].y.values, P[e].p.values
        f, t, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        lw = 2.4 if e == PROPOSED else 1.1
        ax[0].plot(f, t, lw=lw, color=cols[i % 10], label=f"{NAMES.get(e, e)} ({M.auroc(y, p):.3f})")
        ax[1].plot(rc, pr, lw=lw, color=cols[i % 10], label=f"{NAMES.get(e, e)} ({M.auprc(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="1 − Specificity", ylabel="Sensitivity", title="ROC (repeat-averaged OOF)")
    ax[1].axhline(P[exps[0]].y.mean(), color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for a in ax:
        a.legend(fontsize=6.5, loc="lower right" if a is ax[0] else "upper right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_pr.png", dpi=300)
    fig.savefig(FIG / "roc_pr.pdf")
    plt.close(fig)

    # reliability diagram (10 equal-frequency bins) for proposed vs H1 reference vs FIGO-LR
    fig, ax = plt.subplots(figsize=(5, 4.8))
    for e in [x for x in (PROPOSED, H1_REF, "figo_lr_W60") if x in P]:
        y, p = P[e].y.values, P[e].p.values
        o = np.argsort(p)
        bins = np.array_split(o, 10)
        ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-", label=f"{NAMES[e]} (ECE {M.ece(y, p):.3f})")
    m = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, m], [0, m], "k:", lw=0.8)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency (pH<7.05)", title="Reliability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=300)
    fig.savefig(FIG / "reliability.pdf")
    plt.close(fig)

    # decision curve
    th = np.linspace(0.02, 0.30, 57)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, e in enumerate(exps):
        y, p = P[e].y.values, P[e].p.values
        ax.plot(th, M.net_benefit(y, p, th), lw=2.2 if e == PROPOSED else 1.0, color=cols[i % 10], label=NAMES.get(e, e))
    y = P[exps[0]].y.values
    ax.plot(th, y.mean() - (1 - y.mean()) * th / (1 - th), "k--", lw=0.8, label="Treat all")
    ax.axhline(0, color="k", lw=0.8, label="Treat none")
    ax.set(ylim=(-0.02, 0.08), xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "decision_curve.png", dpi=300)
    fig.savefig(FIG / "decision_curve.pdf")
    plt.close(fig)

    # coverage-risk (selective prediction) for ensemble models
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for e in [x for x in (PROPOSED, H1_REF) if x in P]:
        df = load_exp(e)
        for mode, ls in (("unc_epi", "-"), ("unc_conf", "--")):
            covs, risks = [], []
            for _, d in df.groupby("repeat"):
                c, r, _ = M.coverage_risk(d.y.values, d.p.values, d[mode].values, d.thr_s90.values)
                covs.append(c); risks.append(r)
            grid = np.linspace(0.3, 1, 71)
            rr = np.mean([np.interp(grid, c, r) for c, r in zip(covs, risks)], 0)
            ax.plot(grid, rr, ls, label=f"{NAMES[e]} – {'epistemic' if mode == 'unc_epi' else 'confidence'}")
    ax.axvline(0.8, color="gray", lw=0.6)
    ax.set(xlabel="Coverage", ylabel="Risk (error rate, Sp90 threshold)", title="Coverage–risk")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_risk.png", dpi=300)
    fig.savefig(FIG / "coverage_risk.pdf")
    plt.close(fig)


def run():
    TAB.mkdir(parents=True, exist_ok=True)
    main = available([PROPOSED] + FAMILY)
    abl = available(ABL)
    tab = main_table(main + abl)
    tab.to_csv(TAB / "main_metrics.csv", index=False)
    (TAB / "main_metrics.md").write_text(tab.drop(columns="exp").to_markdown(index=False), encoding="utf-8")
    out = {"available": main + abl}
    if PROPOSED in main and H1_REF in main:
        out["H1"] = h1()
        fam = family(refs=[r for r in FAMILY if r in main])
        fam.to_csv(TAB / "family_delong_holm.csv", index=False)
        (TAB / "family_delong_holm.md").write_text(fam.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
        if abl:
            ab = family(refs=abl)
            ab.to_csv(TAB / "ablation_delong.csv", index=False)
        s = get_summary(PROPOSED)
        out["H2"] = s.get("selective")
        out["H2_ref_cnn_ens"] = get_summary(H1_REF).get("selective")
    if PROPOSED in main:
        figures([e for e in main])
    (TAB / "hypotheses.json").write_text(json.dumps(out, indent=1, default=float))
    print(tab.drop(columns="exp").to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "available"}, indent=1, default=float))


if __name__ == "__main__":
    run()
