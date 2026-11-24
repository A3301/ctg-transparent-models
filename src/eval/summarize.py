"""Aggregate a CV experiment (results/cv/<exp>) into metrics with bootstrap CIs.

Primary analysis: repeat-averaged out-of-fold probabilities (one per record).
Threshold metrics: each record classified with the threshold chosen on ITS fold's validation
split, per repeat; reported as the mean over repeats (bootstrap over records).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402


def load_exp(exp: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz")))
    if not files:
        raise FileNotFoundError(exp)
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        r, f = Path(fp).stem[1:].split("_f")
        unc_va = z["unc_epi_va"]
        cut = float(np.quantile(unc_va, 0.80)) if np.ptp(unc_va) > 0 else np.inf
        lv = np.log(np.clip(z["p_va"], 1e-7, 1 - 1e-7) / (1 - np.clip(z["p_va"], 1e-7, 1 - 1e-7)))
        thr = float(z["thr_spec90"])
        lthr = np.log(np.clip(thr, 1e-7, 1 - 1e-7) / (1 - np.clip(thr, 1e-7, 1 - 1e-7)))
        conf_cut = float(np.quantile(-np.abs(lv - lthr), 0.80))
        for i, rec in enumerate(z["records_te"]):
            p = float(z["p_te"][i])
            lt = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
            rows.append(dict(record=str(rec), repeat=int(r), fold=int(f), y=int(z["y_te"][i]), p=p,
                             thr_g=float(z["thr_gmean"]), thr_s90=thr,
                             unc_epi=float(z["unc_epi_te"][i]), unc_ale=float(z["unc_ale_te"][i]),
                             unc_tot=float(z["unc_total_te"][i]), unc_cut=cut,
                             unc_conf=float(-abs(lt - lthr)), conf_cut=conf_cut))
    return pd.DataFrame(rows)


def averaged(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("record").agg(y=("y", "first"), p=("p", "mean"), unc_epi=("unc_epi", "mean")).reset_index()


def _repeat_thr_metric(df, fn, idx=None):
    vals = []
    for r, d in df.groupby("repeat"):
        d = d.set_index("record")
        if idx is not None:
            d = d.loc[idx]
        vals.append(fn(d))
    return float(np.mean(vals))


def summarize(exp: str, n_boot: int = 2000, seed: int = 0) -> dict:
    df = load_exp(exp)
    n_rep = df.repeat.nunique()
    av = averaged(df)
    y, p = av.y.values, av.p.values
    out = {"exp": exp, "n_records": int(len(av)), "n_pos": int(y.sum()), "n_repeats": int(n_rep),
           "n_folds": int(df.groupby("repeat").fold.nunique().min())}
    fns = {"auroc": M.auroc, "auprc": M.auprc, "brier": M.brier, "ece": M.ece,
           "sens_at_spec85": lambda a, b: M.sens_at_spec(a, b, .85),
           "sens_at_spec90": lambda a, b: M.sens_at_spec(a, b, .90),
           "tpr_fpr05": lambda a, b: M.tpr_at_fpr(a, b, .05), "tpr_fpr10": lambda a, b: M.tpr_at_fpr(a, b, .10),
           "tpr_fpr15": lambda a, b: M.tpr_at_fpr(a, b, .15), "tpr_fpr20": lambda a, b: M.tpr_at_fpr(a, b, .20)}
    for k, fn in fns.items():
        est, lo, hi = M.bootstrap_ci(y, p, fn, n_boot, seed)
        out[k] = [est, lo, hi]
    out["cal_slope"], out["cal_intercept"] = M.calibration_slope_intercept(y, p)
    # per-repeat AUROC variability
    rep_auc = [M.auroc(d.y, d.p) for _, d in df.groupby("repeat")]
    out["auroc_per_repeat"] = rep_auc
    # threshold metrics (fold-specific thresholds), mean over repeats + bootstrap over records
    recs = av.record.values
    def thr_fn(col, key):
        return lambda d: M.threshold_metrics(d.y.values, (d.p.values >= d[col].values).astype(float), 0.5)[key]
    rng = np.random.default_rng(seed)
    for col, tag in (("thr_g", "g"), ("thr_s90", "s90")):
        for key in ("se", "sp", "gmean"):
            est = _repeat_thr_metric(df, thr_fn(col, key))
            boots = []
            for _ in range(n_boot // 4):
                idx = recs[rng.integers(0, len(recs), len(recs))]
                if av.set_index("record").loc[idx].y.nunique() < 2:
                    continue
                boots.append(_repeat_thr_metric(df, thr_fn(col, key), idx))
            out[f"{tag}_{key}"] = [est, float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
    out["selective"] = selective(df, n_boot // 4, seed)
    return out


def selective(df: pd.DataFrame, n_boot=500, seed=0) -> dict:
    """H2: defer the most uncertain cases (cut-off from validation -> ~80 % coverage)."""
    if df.unc_epi.abs().max() == 0:
        return {}
    rng = np.random.default_rng(seed)
    recs = df.record.unique()

    def stats(d, mode):
        pred = (d.p.values >= d.thr_s90.values).astype(int)
        if mode == "epi":
            keep = d.unc_epi.values <= d.unc_cut.values
        elif mode == "conf":
            keep = d.unc_conf.values <= d.conf_cut.values
        else:
            keep = np.ones(len(d), bool)
        yk, pk = d.y.values[keep], pred[keep]
        se = (pk[yk == 1] == 1).mean() if (yk == 1).any() else np.nan
        sp = (pk[yk == 0] == 0).mean() if (yk == 0).any() else np.nan
        return np.array([se, sp, keep.mean()])

    def agg(idx, mode):
        res = []
        for _, d in df.groupby("repeat"):
            d = d.set_index("record").loc[idx]
            res.append(stats(d, mode))
        return np.nanmean(res, 0)

    out = {}
    for mode in ("all", "epi", "conf"):
        est = agg(recs, mode)
        out[mode] = dict(se=float(est[0]), sp=float(est[1]), coverage=float(est[2]))
    # random deferral at the realised epistemic coverage (expected value ~ 'all')
    cov = out["epi"]["coverage"]
    rnd = []
    for _ in range(200):
        res = []
        for _, d in df.groupby("repeat"):
            keep = rng.random(len(d)) < cov
            pred = (d.p.values >= d.thr_s90.values).astype(int)
            yk, pk = d.y.values[keep], pred[keep]
            res.append([(pk[yk == 1] == 1).mean(), (pk[yk == 0] == 0).mean()])
        rnd.append(np.nanmean(res, 0))
    rnd = np.array(rnd)
    out["random"] = dict(se=float(rnd[:, 0].mean()), sp=float(rnd[:, 1].mean()), coverage=cov)
    # paired bootstrap of (kept - all) for epistemic deferral
    diffs = []
    for _ in range(n_boot):
        idx = recs[rng.integers(0, len(recs), len(recs))]
        a, e = agg(idx, "all"), agg(idx, "epi")
        diffs.append(e[:2] - a[:2])
    diffs = np.array(diffs)
    for j, k in enumerate(("se", "sp")):
        out[f"diff_{k}_epi_minus_all"] = [float(out["epi"][k] - out["all"][k]),
                                          float(np.nanquantile(diffs[:, j], .025)), float(np.nanquantile(diffs[:, j], .975))]
    return out


if __name__ == "__main__":
    for e in sys.argv[1:]:
        s = summarize(e)
        (ROOT / "results" / "cv" / e / "summary.json").write_text(json.dumps(s, indent=1))
        print(e, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in s.items()
                  if k in ("n_records", "n_pos", "auroc", "auprc", "brier", "ece", "tpr_fpr10", "g_gmean")})
