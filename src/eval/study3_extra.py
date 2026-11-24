"""Additional descriptive reporting for the transparent models (plan dışı #11; TRIPOD+AI items):
sensitivity at fixed specificity, calibration slope / intercept, decision-curve net benefit, and
descriptive ΔAUROC of every transparent model vs the CNN ensemble (95% CI). No hypothesis tests.

Run: python -m src.eval.study3_extra   (with the project .venv)
Output: results/study3/extra_results.json, figures/study3/fig6_calibration_dca.{png,pdf}
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402
from src.eval.study2_eval import N_BOOT, SEED, repeat_avg  # noqa: E402
from src.eval import figstyle as FS  # noqa: E402

MODELS = {"cnn_bayes": "cnn_ens_W60", "figo_lr": "figo_lr_W60", "georgoulas": "georgoulas_W60",
          "transp_lr": "transp_lr_W60", "transp_gam": "transp_gam_W60", "transpdet_gam": "transpdet_gam_W60",
          "transpdetk_gam": "transpdetk_gam_W60",
          # deviation log #16 (descriptive additions)
          "hrv_lr": "hrv_lr_W60", "transp_lr_nohrv": "transp_lr_nohrv_W60", "transp_lr_noq": "transp_lr_noq_W60"}


def main():
    models = {k: v for k, v in MODELS.items() if len(glob.glob(str(ROOT / "results/cv" / v / "r*_f*.npz"))) == 50}
    base = repeat_avg("cnn_ens_W60")
    recs = sorted(base.index)
    y = base.loc[recs].y.values.astype(int)
    P = {k: repeat_avg(v).loc[recs].p.values for k, v in models.items()}
    out = {"models": list(models), "n": len(recs), "n_pos": int(y.sum())}
    th = np.round(np.arange(0.01, 0.301, 0.01), 2)
    for k, p in P.items():
        m = M.all_metrics(y, p)
        out[k] = {kk: m[kk] for kk in ("auroc", "auprc", "sens_at_spec85", "sens_at_spec90", "tpr_fpr10", "brier", "ece",
                                       "cal_slope", "cal_intercept")}
        out[k]["sens_at_spec90_ci"] = M.bootstrap_ci(y, p, lambda a, b: M.sens_at_spec(a, b, 0.90), n=N_BOOT, seed=SEED)
        out[k]["net_benefit"] = dict(zip(th.tolist(), M.net_benefit(y, p, th).tolist()))
        if k != "cnn_bayes":
            d = M.paired_bootstrap_diff(y, p, P["cnn_bayes"], M.auroc, n=N_BOOT, seed=SEED)
            out[k]["auroc_minus_cnn_95ci"] = dict(diff=d[0], lo=d[1], hi=d[2], p=d[3],
                                                  delong_p=M.delong_test(y, p, P["cnn_bayes"])[3])
        # deviation log #16: record-bootstrap CIs for the calibration slope and intercept (fixed predictions)
        out[k]["cal_slope_ci"] = M.bootstrap_ci(y, p, lambda a, b: M.calibration_slope_intercept(a, b)[0], n=N_BOOT, seed=SEED)
        out[k]["cal_intercept_ci"] = M.bootstrap_ci(y, p, lambda a, b: M.calibration_slope_intercept(a, b)[1], n=N_BOOT, seed=SEED)
    # deviation log #16: added value of the 30 variables over the 3 HRV variables, and linear-model ablations
    rev = {}
    for a, b in (("transp_lr", "hrv_lr"), ("hrv_lr", "georgoulas"), ("transp_lr", "transp_lr_nohrv"), ("transp_lr", "transp_lr_noq"),
                 ("hrv_lr", "cnn_bayes")):
        if a in P and b in P:
            d = M.paired_bootstrap_diff(y, P[a], P[b], M.auroc, n=N_BOOT, seed=SEED)
            rev[f"{a}_minus_{b}"] = dict(diff=d[0], lo=d[1], hi=d[2], p=d[3])
    for k in ("hrv_lr", "transp_lr_nohrv", "transp_lr_noq"):      # CI and per-repeat AUROC of the post-hoc models
        if k in P:
            out[k]["auroc_ci"] = M.bootstrap_ci(y, P[k], M.auroc, n=N_BOOT, seed=SEED)
            reps = []
            for r in range(5):
                rows = []
                for fp in sorted(glob.glob(str(ROOT / "results/cv" / MODELS[k] / f"r{r}_f*.npz"))):
                    z = np.load(fp, allow_pickle=True)
                    rows += list(zip(z["y_te"], z["p_te"]))
                yy, pp = np.array([a for a, _ in rows]), np.array([b for _, b in rows])
                reps.append(M.auroc(yy, pp))
            out[k]["per_repeat"] = [float(np.mean(reps)), float(np.std(reps, ddof=1))]
    out["review_additions"] = rev
    out["treat_all_net_benefit"] = dict(zip(th.tolist(), (y.mean() - (1 - y.mean()) * th / (1 - th)).tolist()))
    avg = (P["transp_lr"] + P["cnn_bayes"]) / 2
    d = M.paired_bootstrap_diff(y, avg, P["transp_lr"], M.auroc, n=N_BOOT, seed=SEED)
    out["avg_transp_lr_cnn"] = dict(auroc=M.auroc(y, avg), minus_transp_lr=d)
    (ROOT / "results/study3/extra_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # calibration / decision-curve figure: reliability by deciles + decision curve, journal artwork style
    FS.setup()
    show = [("cnn_bayes", "cnn"), ("transp_lr", "transp_lr"), ("transp_gam", "transp_gam")]
    fig, ax = plt.subplots(1, 2, figsize=(FS.WIDTH, 2.7), gridspec_kw={"wspace": 0.3})
    ax[0].plot([0, 0.4], [0, 0.4], color=FS.INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax[0].text(0.33, 0.355, "ideal", rotation=45, fontsize=7, color=FS.INK2, ha="center", va="center")
    handles = []
    for k, key in show:
        name, col, mk = FS.MODEL[key]
        p = P[k]
        o = np.argsort(p)
        bins = np.array_split(o, 10)
        h, = ax[0].plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "-" + mk, color=col, lw=1.0, ms=3.8,
                        mec="white", mew=0.5, zorder=3, label=name)
        handles.append(h)
    ax[0].set_xlim(0, 0.4)
    ax[0].set_ylim(0, 0.4)
    ax[0].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax[0].set_xlabel("Predicted risk (decile mean)")
    ax[0].set_ylabel("Observed proportion, pH < 7.05")
    FS.value_grid(ax[0], "both")
    FS.panel_title(ax[0], "A", "Calibration")
    ax[1].axhline(0, color=FS.INK2, lw=0.8, zorder=1)
    ax[1].text(0.298, -0.0015, "treat none", fontsize=7, color=FS.INK2, ha="right", va="top")
    hT, = ax[1].plot(th, list(out["treat_all_net_benefit"].values()), color=FS.INK2, lw=1.0, ls=(0, (3, 2)), label="Treat all")
    for k, key in show:
        name, col, mk = FS.MODEL[key]
        ax[1].plot(th, list(out[k]["net_benefit"].values()), color=col, lw=1.4, zorder=3)
    ax[1].set_xlim(0, 0.305)
    ax[1].set_ylim(-0.01, 0.07)
    ax[1].set_xlabel("Threshold probability")
    ax[1].set_ylabel("Net benefit")
    FS.value_grid(ax[1], "y")
    FS.panel_title(ax[1], "B", "Decision curve")
    fig.legend(handles=handles + [hT], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    FS.save(fig, ROOT / "figures/study3", "fig6_calibration_dca")
    for k in models:
        r = out[k]
        print(f"{k:15s} auc {r['auroc']:.3f} se@sp90 {r['sens_at_spec90']:.3f} slope {r['cal_slope']:.2f} itc {r['cal_intercept']:.2f} "
              f"brier {r['brier']:.4f}" + (f" d_cnn {r['auroc_minus_cnn_95ci']['diff']:+.3f} [{r['auroc_minus_cnn_95ci']['lo']:+.3f},{r['auroc_minus_cnn_95ci']['hi']:+.3f}]" if k != "cnn_bayes" else ""))


if __name__ == "__main__":
    main()
