"""Evaluation metrics, bootstrap CIs, DeLong test, Holm correction, selective prediction."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc(y, p):
    return float(roc_auc_score(y, p))


def auprc(y, p):
    return float(average_precision_score(y, p))


def sens_at_spec(y, p, spec):
    y, p = np.asarray(y), np.asarray(p)
    thr = np.quantile(p[y == 0], spec, method="higher")
    return float((p[y == 1] > thr).mean())


def tpr_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_metrics(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    se = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    g = float(np.sqrt(se * sp))
    # QI in the CTG literature = sqrt(Se*Sp) (verified on Zhao 2019 / Georgoulas 2017), i.e. = G-mean
    return dict(se=se, sp=sp, ppv=ppv, gmean=g, qi=g, tp=tp, fn=fn, tn=tn, fp=fp)


def gmean_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    g = np.sqrt(tpr * (1 - fpr))
    return float(thr[int(np.argmax(g))])


def spec_threshold(y, p, spec=0.90):
    p0 = np.asarray(p)[np.asarray(y) == 0]
    return float(np.quantile(p0, spec, method="higher"))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-frequency bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def calibration_slope_intercept(y, p):
    import statsmodels.api as sm
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        m = sm.GLM(np.asarray(y), sm.add_constant(lp), family=sm.families.Binomial()).fit()
        slope = float(m.params[1])
        itc = sm.GLM(np.asarray(y), np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
        return slope, float(itc.params[0])
    except Exception:
        return np.nan, np.nan


def net_benefit(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def all_metrics(y, p, thr_g=None, thr_s90=None):
    y, p = np.asarray(y), np.asarray(p)
    m = dict(auroc=auroc(y, p), auprc=auprc(y, p),
             sens_at_spec85=sens_at_spec(y, p, 0.85), sens_at_spec90=sens_at_spec(y, p, 0.90),
             tpr_fpr05=tpr_at_fpr(y, p, 0.05), tpr_fpr10=tpr_at_fpr(y, p, 0.10),
             tpr_fpr15=tpr_at_fpr(y, p, 0.15), tpr_fpr20=tpr_at_fpr(y, p, 0.20),
             brier=brier(y, p), ece=ece(y, p))
    m["cal_slope"], m["cal_intercept"] = calibration_slope_intercept(y, p)
    if thr_g is not None:
        tm = threshold_metrics(y, p, thr_g)
        m.update({f"g_{k}": v for k, v in tm.items() if k in ("se", "sp", "gmean", "qi", "ppv")})
    return m


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(y, p, fn, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    vals = np.array(vals)
    return float(fn(y, p)), float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y, pa, pb, fn, n=2000, seed=0, alpha=0.05):
    """fn(y, pa) - fn(y, pb) with percentile CI and two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    d = np.array(d)
    obs = fn(y, pa) - fn(y, pb)
    pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(np.nanquantile(d, alpha / 2)), float(np.nanquantile(d, 1 - alpha / 2)), float(min(pval, 1.0))


# ---------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y, pa, pb):
    """Paired DeLong test for AUROC(pa) - AUROC(pb). Returns (diff, se, z, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    y = y[order]
    preds = np.vstack([np.asarray(pa)[order], np.asarray(pb)[order]])
    m = int(y.sum())
    n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    se = float(np.sqrt(max(var, 1e-12)))
    z = diff / se
    return float(diff), se, float(z), float(2 * stats.norm.sf(abs(z)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------- selective prediction
def coverage_risk(y, p, unc, thr):
    """Risk = misclassification rate at threshold thr among retained cases, ordered by certainty."""
    y, p, unc = map(np.asarray, (y, p, unc))
    order = np.argsort(unc)
    err = ((p[order] >= thr).astype(int) != y[order]).astype(float)
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    return cov, risk, float(np.trapezoid(risk, cov))


def selective_metrics(y, p, unc, thr, unc_cut):
    y, p, unc = map(np.asarray, (y, p, unc))
    keep = unc <= unc_cut
    out = dict(coverage=float(keep.mean()))
    tm_all = threshold_metrics(y, p, thr)
    tm_k = threshold_metrics(y[keep], p[keep], thr) if keep.any() else {}
    out.update({f"all_{k}": tm_all[k] for k in ("se", "sp")})
    out.update({f"kept_{k}": tm_k.get(k, np.nan) for k in ("se", "sp")})
    out["kept_n_pos"] = int(y[keep].sum())
    out["deferred_n_pos"] = int(y[~keep].sum())
    if keep.any() and y[keep].min() != y[keep].max():
        out["kept_auroc"] = auroc(y[keep], p[keep])
    return out
