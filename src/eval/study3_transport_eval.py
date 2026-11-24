"""Evaluation of Plan C transportability runs (docs/analiz_plani_v3_ek3.md); written before results.

Run: python -m src.eval.study3_transport_eval
Output: results/study3/transport_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402

MODELS = ["transp_lr", "transp_gam", "figo_lr", "georgoulas", "cnn"]
N_BOOT, SEED = 2000, 20261123


def main():
    out = {}
    for split in ("C1", "C3"):
        P, y = {}, None
        for m in MODELS:
            fp = ROOT / "results" / "transport" / f"{split}_{m}.npz"
            if not fp.exists():
                continue
            z = np.load(fp, allow_pickle=True)
            yy = z["y_te"].astype(int)
            assert y is None or (yy == y).all()
            y, P[m] = yy, z["p_te"]
            out.setdefault(split, {"n_test": int(len(yy)), "events_test": int(yy.sum()),
                                   "n_train": int(z["n_train"]), "events_train": int(z["events_train"])})
        for m, p in P.items():
            s, i = M.calibration_slope_intercept(y, p)
            out[split][m] = dict(auroc=M.bootstrap_ci(y, p, M.auroc, n=N_BOOT, seed=SEED), brier=M.brier(y, p),
                                 cal_slope=s, cal_intercept=i)
        if "cnn" in P:
            for m in ("transp_lr", "transp_gam"):
                if m in P:
                    out[split][f"{m}_minus_cnn"] = M.paired_bootstrap_diff(y, P[m], P["cnn"], M.auroc, n=N_BOOT, seed=SEED)
    p = ROOT / "results" / "study3" / "transport_results.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for split, r in out.items():
        print(split, {k: r[k] for k in ("n_train", "events_train", "n_test", "events_test")})
        for m in MODELS:
            if m in r:
                print(f"  {m:11s} AUROC {r[m]['auroc'][0]:.3f} [{r[m]['auroc'][1]:.3f},{r[m]['auroc'][2]:.3f}] "
                      f"slope {r[m]['cal_slope']:.2f} itc {r[m]['cal_intercept']:.2f}")
        for m in ("transp_lr", "transp_gam"):
            k = f"{m}_minus_cnn"
            if k in r:
                print(f"  {k}: {r[k][0]:+.3f} [{r[k][1]:+.3f},{r[k][2]:+.3f}]")


if __name__ == "__main__" and "--ek4" not in sys.argv:
    main()


# ---------------------------------------------------------------- ek4 (Plan B recalibration, Plan C UC-free models)
def recalibrate_loo(y, p, slope=False):
    """Leave-one-out logistic recalibration on the logit scale within the target group."""
    import statsmodels.api as sm
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        if slope:
            f = sm.GLM(y[m], sm.add_constant(lp[m]), family=sm.families.Binomial()).fit()
            out[i] = f.params[0] + f.params[1] * lp[i]
        else:
            f = sm.GLM(y[m], np.ones(m.sum()), family=sm.families.Binomial(), offset=lp[m]).fit()
            out[i] = f.params[0] + lp[i]
    return 1 / (1 + np.exp(-out))


def ek4():
    res = {}
    models = MODELS + ["transp_lr_nouc", "transp_gam_nouc"]
    for split in ("C1", "C3"):
        P, y = {}, None
        for m in models:
            fp = ROOT / "results" / "transport" / f"{split}_{m}.npz"
            if fp.exists():
                z = np.load(fp, allow_pickle=True)
                y, P[m] = z["y_te"].astype(int), z["p_te"]
        r = {}
        for m, p in P.items():
            row = {"auroc": M.bootstrap_ci(y, p, M.auroc, n=N_BOOT, seed=SEED), "brier_raw": M.brier(y, p)}
            for nm, sl in (("intercept", False), ("intercept_slope", True)):
                q = recalibrate_loo(y, p, sl)
                s_, i_ = M.calibration_slope_intercept(y, q)
                row[f"recal_{nm}"] = dict(brier=M.brier(y, q), cal_slope=s_, cal_intercept=i_)
            r[m] = row
        for m in ("transp_lr_nouc", "transp_gam_nouc"):
            if m in P and "cnn" in P:
                r[f"{m}_minus_cnn"] = M.paired_bootstrap_diff(y, P[m], P["cnn"], M.auroc, n=N_BOOT, seed=SEED)
                base = m.replace("_nouc", "")
                r[f"{m}_minus_{base}"] = M.paired_bootstrap_diff(y, P[m], P[base], M.auroc, n=N_BOOT, seed=SEED)
        res[split] = r
    # within-CV performance of UC-free models (main protocol)
    from src.eval.study2_eval import repeat_avg
    cnn = repeat_avg("cnn_ens_W60")
    recs = sorted(cnn.index)
    yy = cnn.loc[recs].y.values
    for m in ("transp_lr_nouc", "transp_gam_nouc"):
        p = repeat_avg(f"{m}_W60").loc[recs].p.values
        res[f"cv_{m}"] = dict(auroc=M.bootstrap_ci(yy, p, M.auroc, n=N_BOOT, seed=SEED),
                              minus_cnn=M.paired_bootstrap_diff(yy, p, cnn.loc[recs].p.values, M.auroc, n=N_BOOT, seed=SEED))
    (ROOT / "results" / "study3" / "transport_ek4_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__" and "--ek4" in sys.argv:
    r = ek4()
    for k, v in r.items():
        if k.startswith("cv_"):
            print(k, [round(x, 3) for x in v["auroc"]], "minus_cnn", [round(x, 3) for x in v["minus_cnn"]])
        else:
            for m, row in v.items():
                if isinstance(row, dict):
                    print(k, m, "AUROC", round(row["auroc"][0], 3), "Brier raw", round(row["brier_raw"], 4),
                          "recal(int)", round(row["recal_intercept"]["brier"], 4), "itc", round(row["recal_intercept"]["cal_intercept"], 2),
                          "recal(int+sl)", round(row["recal_intercept_slope"]["brier"], 4))
                else:
                    print(k, m, [round(x, 3) for x in row])
