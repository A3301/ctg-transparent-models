"""Nested cross-validation runner (frozen protocol, docs/analiz_plani.md section 4).

usage: python -m src.run.cv --model cnn --window W60 --label y_ph705 --splits outer_splits \
           --exp cnn_W60 [--ensemble 5] [--workers 4] [--repeats 0,1,2,3,4]

Each outer fold result is saved to results/cv/<exp>/r{r}_f{f}.npz (resumable).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SEED = 20261122
N_INNER, K_DL, K_SK, VAL_FRAC = 3, 6, 30, 0.15


# ------------------------------------------------------------------ search spaces
def sample_cfg(model, rng):
    c = lambda xs: xs[rng.integers(len(xs))]  # noqa: E731
    base = dict(lr=c([1e-3, 3e-4]), wd=c([1e-4, 1e-3, 1e-2]), drop=c([0.1, 0.3, 0.5]), bs=32)
    if model in ("cnn", "nesy") or model.startswith("nesy"):
        base.update(width=c([16, 32, 48]), n_stages=c([3, 4]), kernel=c([5, 7, 9]))
    if model.startswith("nesy"):
        base.update(lam=c([0.05, 0.1, 0.3]))
    if model == "gru":
        base.update(hidden=c([32, 64]), layers=c([1, 2]), pool=4)
    if model == "transformer":
        base.update(d=c([32, 64]), layers=c([2, 3]), heads=4, patch=c([30, 60]))
    if model in ("mcnn", "fhrlinet"):
        base.update(lr=c([1e-4, 3e-4, 1e-3]))
    return base


def default_first(model, cfgs):
    """Put the paper's published configuration first for re-implemented methods."""
    if model == "fhrlinet":
        cfgs[0] = dict(lr=1e-4, wd=0.0, drop=0.5, bs=32)
    if model == "mcnn":
        cfgs[0] = dict(lr=1e-4, wd=0.0, drop=0.3, bs=32)
    return cfgs


def sample_sk(model, rng):
    if model == "figo_lr":
        return dict(C=float(10 ** rng.uniform(-3, 2)))
    if model == "figo_xgb":
        return dict(max_depth=int(rng.integers(2, 5)), n_estimators=int(rng.integers(50, 301)),
                    learning_rate=float(10 ** rng.uniform(-1.5, -0.5)), subsample=float(rng.uniform(0.6, 1)),
                    colsample_bytree=float(rng.uniform(0.6, 1)), min_child_weight=float(rng.uniform(1, 10)),
                    reg_lambda=float(10 ** rng.uniform(-1, 1)))
    if model == "georgoulas":
        return dict(C=float(10 ** rng.uniform(-2, 2)), gamma=float(10 ** rng.uniform(-3, 0)))
    if model in TRANSP_GAM:                     # analiz_plani_v3 §3.2 / §4.2
        return dict(C=float(10 ** rng.uniform(-3, 1)), n_knots=int(rng.choice([3, 4, 5])))
    if model in TRANSP_LR:
        return dict(C=float(10 ** rng.uniform(-3, 2)))
    raise ValueError(model)


# ------------------------------------------------------------------ end-to-end model (analiz_plani_v2)
NESY2 = ("nesy2", "nesy2_nomask", "det", "detk")      # det: Plan B detector (analiz_plani_v3 §4.1)
Q_REGION = ("nesy2", "det", "detk")
DETECTORS = ("det", "detk")                    # v3 Plan B / Plan E (ek2)                     # models supervised / evaluated in the q = 1 region
EXPERT_AUX = ("frame", "fmask", "ftype", "wexp")      # Romagnoli-derived targets (blanked for test rows)


def nesy2_targets(model, window, recs):
    """Expert targets + quality mask (§3) and weak labels for concepts without expert reference."""
    assert window == "W60", "the end-to-end model analysis is W60 only"
    from src.data.concept_targets import load_targets
    T = load_targets()
    assert list(T["records"]) == list(recs), "expert targets not aligned with records"
    frame = np.stack([T["frame_acc"], T["frame_dec"], T["frame_uc"]], 1).astype(np.uint8)
    region = T["q"] if model in Q_REGION else T["cov"]          # no-mask ablation: every covered bin known
    fmask = ((frame == 1) | (region[:, None, :] == 1)).astype(np.uint8)
    wexp = T["win_masked"] if model in Q_REGION else T["win_nomask"]
    t = __import__("src.data.inputs", fromlist=["figo_table"]).figo_table(window).reindex(recs)
    aux = {"frame": frame, "fmask": fmask, "ftype": T["frame_dtype"].astype(np.int64), "wexp": wexp.astype(np.int64)}
    for c in ("c_baseline", "c_variability", "c_reduced_var_long", "c_increased_var_long"):
        v = t[c].values.astype(float)
        aux[c] = np.where(np.isfinite(v), v, -1).astype(np.int64)
    return aux


def blank_test_targets(extra, te):
    """§7: outer-test records' expert targets must not exist during training/tuning/calibration."""
    a = extra.get("__aux__")
    if a is None or "frame" not in a:
        return
    a = {k: v.copy() for k, v in a.items()}
    a["frame"][te] = 0
    a["fmask"][te] = 0
    a["ftype"][te] = -1
    a["wexp"][te] = -1
    assert not a["fmask"][te].any() and (a["ftype"][te] < 0).all() and (a["wexp"][te] < 0).all()
    extra["__aux__"] = a


# ------------------------------------------------------------------ data per model
def pd_index(recs):
    import pandas as pd
    return pd.Index(recs, name="record")


TRANSP_GAM = ("transp_gam", "transp_gam_noq", "transp_gam_nohrv", "transpdet_gam", "transpdetk_gam", "transp_gam_nouc")
# linear models; the _nohrv/_noq ablations and the HRV-only linear model were added after the primary results (deviation log #16)
TRANSP_LR = ("transp_lr", "transp_lr_nouc", "transp_lr_nohrv", "transp_lr_noq", "hrv_lr")
TRANSP = TRANSP_GAM + TRANSP_LR
UC_FEATURES = ["uc_per_10min", "late_frac_contr", "variable_frac_contr", "early_frac_contr", "any_dec_frac_contr",
               "uc_missing", "uc_longest_gap_s"]   # analiz_plani_v3_ek4 Plan C
QUALITY_FEATURES = ["fhr_missing", "uc_missing", "fhr_longest_gap_s", "uc_longest_gap_s"]
HRV_FEATURES = ["vlf", "lf", "sd2"]


def baseline_deviation(window, recs):
    """Plan B input channel: (FHR - rule-based baseline curve) / 25 bpm at 1 Hz, 0 where invalid.
    Signal-only (no outcome, no expert labels); cached under data/processed (git-ignored)."""
    cache = ROOT / "data" / "processed" / f"basedev_{window}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        assert list(z["records"]) == list(recs)
        return z["dev"]
    from joblib import Parallel, delayed
    from src.data.inputs import _down, _load
    from src.features.figo import baseline_curve
    d = _load(window)
    assert list(d["records"].astype(str)) == list(recs)

    def one(i):
        v = d["fhr_mask"][i] > 0.5
        b = baseline_curve(d["fhr"][i], v)
        return np.where(v & np.isfinite(b), d["fhr"][i] - b, 0.0), (v & np.isfinite(b)).astype(np.float32)
    res = Parallel(n_jobs=12)(delayed(one)(i) for i in range(len(recs)))
    dev4, m4 = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
    dev, _ = _down(dev4, m4, 4)
    dev = (dev / 25.0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, records=np.array(recs), dev=dev)
    return dev


def rule_event_indicators(fhr, fhr_mask, uc, uc_mask):
    """1 Hz 0/1 indicators of the rule-based ACC, DEC and UC events of one 4 Hz window (signal only)."""
    from src.features.figo import baseline_curve, detect_contractions, detect_events
    v, uv = fhr_mask > 0.5, uc_mask > 0.5
    b = baseline_curve(fhr, v)
    m = np.zeros((3, len(fhr)), np.float32)
    for k, evs in enumerate((detect_events(fhr, v, b, +1), detect_events(fhr, v, b, -1), detect_contractions(uc, uv)[0])):
        for e in evs:
            m[k, e["start"]:e["end"]] = 1.0
    return m.reshape(3, -1, 4).max(-1)


def rule_event_channels(window, recs):
    cache = ROOT / "data" / "processed" / f"ruleev_{window}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        assert list(z["records"]) == list(recs)
        return z["ev"]
    from joblib import Parallel, delayed
    from src.data.inputs import _load
    d = _load(window)
    assert list(d["records"].astype(str)) == list(recs)
    ev = np.stack(Parallel(n_jobs=12)(delayed(rule_event_indicators)(d["fhr"][i], d["fhr_mask"][i], d["uc"][i],
                                                                     d["uc_mask"][i]) for i in range(len(recs))))
    np.savez(cache, records=np.array(recs), ev=ev.astype(np.float32))
    return ev.astype(np.float32)


def transp_table(model, window):
    """Plan C inputs (analiz_plani_v3 §3.1): FIGO features + HRV + signal quality, aligned by record."""
    import pandas as pd
    from src.data.inputs import FIGO_FEATURES, figo_table
    f = figo_table(window)[FIGO_FEATURES]
    h = pd.read_csv(ROOT / "results" / "features" / f"hrv_{window}.csv", dtype={"record": str}).set_index("record")
    q = pd.read_csv(ROOT / "results" / "preprocess" / "window_quality.csv", dtype={"record": str})
    q = q[q.window == window].set_index("record")[QUALITY_FEATURES]
    parts = [f]
    if model == "hrv_lr":                       # HRV-only linear model (deviation log #16)
        return h.reindex(f.index)[HRV_FEATURES]
    if not model.endswith("_nohrv"):
        parts.append(h[HRV_FEATURES])
    if not model.endswith("_noq"):
        parts.append(q)
    t = pd.concat(parts, axis=1, join="inner")
    assert len(t) == len(f), "feature tables not aligned"
    if model.endswith("_nouc"):
        t = t.drop(columns=UC_FEATURES)
    return t


def model_inputs(model, window):
    """Return records, X, extra(dict of per-record arrays), hz, kind."""
    from src.data.inputs import FIGO_FEATURES, check_no_forbidden, figo_table, quality_vector, signal_inputs  # noqa: F401
    if model in ("figo_lr", "figo_xgb"):
        t = figo_table(window)
        check_no_forbidden(FIGO_FEATURES)
        return t.index.values.astype(str), t[FIGO_FEATURES].values.astype(np.float32), {}, None, "sk"
    if model in TRANSP:
        t = transp_table(model, window)
        check_no_forbidden(t.columns)
        return t.index.values.astype(str), t.values.astype(np.float32), {}, None, "sk"
    if model == "georgoulas":
        p = ROOT / "results" / "features" / f"hrv_{window}.csv"
        import pandas as pd
        t = pd.read_csv(p, dtype={"record": str}).set_index("record")
        check_no_forbidden(t.columns)
        return t.index.values.astype(str), t[["vlf", "lf", "sd2"]].values.astype(np.float32), {}, None, "sk"
    hz = 0.25 if model in ("mcnn", "fhrlinet") else 1.0
    recs, X = signal_inputs(window, hz)
    extra = {}
    if model == "mcnn":
        X = X.copy()
        poor = X[:, 3].mean(1) < 0.5          # poor UC quality -> zeros (Petrozziello 2019)
        X[poor, 2] = 0.0
        extra["q"] = quality_vector(window)
    if model in DETECTORS:
        X = np.concatenate([X, baseline_deviation(window, recs)[:, None, :]], 1).astype(np.float32)
    if model == "detk":     # ek2: rule-based ACC / DEC / UC event indicators as extra input channels
        X = np.concatenate([X, rule_event_channels(window, recs)], 1).astype(np.float32)
    if model in NESY2:
        extra["__aux__"] = nesy2_targets(model, window, recs)
        return recs, X, extra, hz, "dl"
    if model.startswith("nesy"):
        from src.data.inputs import CONCEPTS
        from src.models.nesy import concept_vector
        t = figo_table(window).reindex(recs)
        assert t.index.equals(pd_index(recs)), "concept table not aligned with records"
        if model == "nesy_rulesonly":
            extra["cvec"] = concept_vector(t)
        if model == "nesy_clin":     # FLAGGED clinical-prior experiment (pre-labour variables only)
            import pandas as pd
            from src.data.ctu_uhb import CLINICAL_PRIOR_FIELDS
            meta = pd.read_csv(ROOT / "results/eda/metadata_all.csv", dtype={"record": str}).set_index("record").reindex(recs)
            c = pd.DataFrame(index=meta.index)
            c["gest_ge41"] = meta.gest_weeks >= 41
            c["mat_age_ge35"] = meta.mat_age >= 35
            c["nulliparous"] = meta.parity == 0
            for k in ("diabetes", "hypertension", "preeclampsia", "liq_praecox", "pyrexia", "induced"):
                c[k] = meta[k] > 0
            assert set(["gest_weeks", "mat_age", "parity", "diabetes", "hypertension", "preeclampsia",
                        "liq_praecox", "pyrexia", "induced"]) <= set(CLINICAL_PRIOR_FIELDS)
            check_no_forbidden(c.columns)
            extra["clin"] = c.fillna(False).values.astype(np.float32)
        extra_aux = {}
        for cname in CONCEPTS:
            v = t[cname].values.astype(float)
            extra_aux[cname] = np.where(np.isfinite(v), v, -1).astype(np.int64)
        extra["__aux__"] = extra_aux
    return recs, X, extra, hz, "dl"


def build_net(model, cfg, X, extra):
    from src.models.nets import REGISTRY
    if model in DETECTORS:
        from src.models.det import DetNet
        kw = {k: v for k, v in cfg.items() if k in ("width", "n_stages", "kernel", "drop")}
        return DetNet(X.shape[1], **kw)
    if model in NESY2:
        from src.models.nesy_v2 import NeSyV2
        kw = {k: v for k, v in cfg.items() if k in ("width", "n_stages", "kernel", "drop")}  # lam ignored (fixed weights)
        return NeSyV2(X.shape[1], **kw)
    if model.startswith("nesy"):
        from src.models.nesy import build_nesy
        return build_nesy(model, cfg, in_ch=X.shape[1])
    kw = dict(cfg)
    if model == "mcnn":
        kw.update(length=X.shape[-1], n_q=extra["q"].shape[1])
    if model == "transformer":
        kw.update(max_len=X.shape[-1])
    return REGISTRY[model](in_ch=X.shape[1], **kw)


# ------------------------------------------------------------------ one outer fold
def _split_extra(extra, idx):
    return {k: v[idx] for k, v in extra.items() if k != "__aux__"}


def _split_aux(extra, idx):
    a = extra.get("__aux__")
    return None if a is None else {k: v[idx] for k, v in a.items()}


def fit_predict_dl(model, cfg, X, y, extra, tr, te, seed, device, n_members=1, return_members=False,
                   save_prefix=None):
    """Train on tr (with stratified val carve-out), calibrate on val, predict te."""
    from sklearn.model_selection import train_test_split
    from src.train.trainer import calibrated_probs, train_model
    fit, val = train_test_split(tr, test_size=VAL_FRAC, stratify=y[tr], random_state=seed)
    crop = (30 * 60 // 4, 5 * 60 // 4) if model == "fhrlinet" else None
    mem_te, mem_va, temps, epochs, extras_te, rule_w, fval = [], [], [], [], [], [], []
    for m in range(n_members):
        net = build_net(model, cfg, X, extra)
        net = train_model(net, X[fit], y[fit], X[val], y[val], cfg,
                          _split_extra(extra, fit), _split_extra(extra, val),
                          _split_aux(extra, fit), _split_aux(extra, val),
                          seed=seed + 7919 * m, device=device, crop=crop)
        pt, pv, T, ot = calibrated_probs(net, X[te], _split_extra(extra, te), X[val], y[val],
                                         _split_extra(extra, val), device)
        mem_te.append(pt); mem_va.append(pv); temps.append(T); epochs.append(net.epochs_)
        extras_te.append({k: v for k, v in ot.items() if k not in ("logit", "f_logit", "f_type")})
        if model in NESY2:   # validation frame probabilities for event-threshold selection (§4.1)
            from src.train.trainer import predict_logits
            fval.append(predict_logits(net, X[val], _split_extra(extra, val), device)["f_prob"])
        if save_prefix is not None:   # final outer-fold members only (needed for the FHRMA check, §6.5)
            import torch
            Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": net.state_dict(), "cfg": cfg, "T": T, "pos_weight": net.pos_weight_},
                       f"{save_prefix}_m{m}.pt")
        if hasattr(net, "logic"):
            rule_w.append(net.logic.rule_weights().detach().cpu().numpy())
        del net
        import torch
        torch.cuda.empty_cache()
    return dict(val_idx=val, mem_te=np.array(mem_te), mem_va=np.array(mem_va), temps=temps,
                epochs=epochs, extras_te=extras_te, rule_w=np.array(rule_w),
                fval=np.mean(fval, 0) if fval else None)


def fit_predict_sk(model, cfg, X, y, tr, te, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from src.train.trainer import fit_temperature
    fit, val = train_test_split(tr, test_size=VAL_FRAC, stratify=y[tr], random_state=seed)
    pw = (len(fit) - y[fit].sum()) / max(y[fit].sum(), 1)
    if model == "figo_lr":
        from sklearn.linear_model import LogisticRegression
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000))
    elif model in TRANSP_GAM or model in TRANSP_LR:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import SplineTransformer
        steps = [SimpleImputer(strategy="median"), StandardScaler()]
        if model in TRANSP_GAM:     # additive model: per-feature cubic B-splines, no interactions
            steps.append(SplineTransformer(n_knots=cfg["n_knots"], degree=3, knots="quantile",
                                           extrapolation="linear"))
        clf = make_pipeline(*steps, LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000))
    elif model == "figo_xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**cfg, scale_pos_weight=pw, eval_metric="logloss", n_jobs=2,
                            random_state=seed, verbosity=0)
    else:
        from sklearn.svm import SVC
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            SVC(C=cfg["C"], gamma=cfg["gamma"], class_weight="balanced"))
    clf.fit(X[fit], y[fit])

    def score(idx):
        if model == "georgoulas":
            return clf.decision_function(X[idx])
        p = np.clip(clf.predict_proba(X[idx])[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)) - math.log(pw)   # prior correction (balanced weights)
    sv, st = score(val), score(te)
    if model == "georgoulas":   # Platt scaling (decision function is not a logit)
        from sklearn.linear_model import LogisticRegression
        pl = LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), y[val])
        pv, pt = pl.predict_proba(sv.reshape(-1, 1))[:, 1], pl.predict_proba(st.reshape(-1, 1))[:, 1]
        T = float("nan")
    else:
        T = fit_temperature(sv, y[val])
        pv, pt = 1 / (1 + np.exp(-sv / T)), 1 / (1 + np.exp(-st / T))
    return dict(val_idx=val, mem_te=pt[None], mem_va=pv[None], temps=[T], epochs=[0], extras_te=[{}])


def inner_select(model, kind, X, y, extra, tr, seed, device):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    if kind == "dl":
        cfgs = default_first(model, [sample_cfg(model, rng) for _ in range(K_DL)])
    else:
        cfgs = [sample_sk(model, rng) for _ in range(K_SK)]
    skf = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    scores = []
    for cfg in cfgs:
        s = []
        for itr, ite in skf.split(tr, y[tr]):
            a, b = tr[itr], tr[ite]
            if kind == "dl":
                r = fit_predict_dl(model, cfg, X, y, extra, a, b, seed, device, 1)
            else:
                r = fit_predict_sk(model, cfg, X, y, a, b, seed)
            if model in DETECTORS:     # v3 §4.1: select by mean ACC/DEC event F1 (best grid threshold), not outcome
                from src.data.concept_targets import load_targets
                from src.eval.events import select_thresholds
                _, curves = select_thresholds(r["extras_te"][0]["f_prob"], b, load_targets(), "q")
                s.append(float(np.mean([np.nanmax(np.array(curves[k], float)) for k in (0, 1)])))
            else:
                s.append(roc_auc_score(y[b], r["mem_te"].mean(0)))
        scores.append(float(np.mean(s)))
    best = int(np.argmax(scores))
    return cfgs[best], cfgs, scores


def run_fold(args_tuple):
    (model, window, label, splits_name, exp, r, f, n_ens, device, fixed_cfg) = args_tuple
    out = ROOT / "results" / "cv" / exp / f"r{r}_f{f}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return str(out), "skip"
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        torch.set_num_threads(2)
        from src.data.inputs import labels
        from src.eval.metrics import gmean_threshold, spec_threshold
        splits = json.loads((ROOT / "results" / "splits" / f"{splits_name}.json").read_text())
        fold = splits["repeats"][r][f]
        recs, X, extra, hz, kind = model_inputs(model, window)
        pos = {k: i for i, k in enumerate(recs)}
        y_all = labels(label).reindex(recs).values
        tr = np.array([pos[k] for k in fold["train"]])
        te = np.array([pos[k] for k in fold["test"]])
        y = np.nan_to_num(y_all, nan=0).astype(np.int64)
        seed = SEED + 1000 * r + f
        if model in NESY2:
            blank_test_targets(extra, te)
        t0 = time.time()
        if fixed_cfg is not None and fixed_cfg.startswith("mode:"):   # sensitivity: most frequent primary config
            import glob as _g
            from collections import Counter
            cs = [str(np.load(fp, allow_pickle=True)["cfg"]) for fp in
                  sorted(_g.glob(str(ROOT / "results" / "cv" / fixed_cfg[5:] / "r*_f*.npz")))]
            cfg, cfgs, scores = json.loads(Counter(cs).most_common(1)[0][0]), [], []
        elif fixed_cfg is not None:                   # ablations reuse the proposed model's per-fold config
            ref = np.load(ROOT / "results" / "cv" / fixed_cfg / f"r{r}_f{f}.npz", allow_pickle=True)
            cfg, cfgs, scores = json.loads(str(ref["cfg"])), [], []
            if model.startswith("nesy"):
                cfg.update(json.loads(str(ref["cfg"])))
        else:
            cfg, cfgs, scores = inner_select(model, kind, X, y, extra, tr, seed, device)
        if kind == "dl":
            sp = str(out.parent / "weights" / f"r{r}_f{f}") if model in NESY2 else None
            res = fit_predict_dl(model, cfg, X, y, extra, tr, te, seed, device, n_ens, save_prefix=sp)
        else:
            res = fit_predict_sk(model, cfg, X, y, tr, te, seed)
        val = res["val_idx"]
        p_va, p_te = res["mem_va"].mean(0), res["mem_te"].mean(0)

        def ent(p):
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))
        tot_te, ale_te = ent(p_te), ent(res["mem_te"]).mean(0)
        tot_va, ale_va = ent(p_va), ent(res["mem_va"]).mean(0)
        extras = {}
        for k in res["extras_te"][0]:
            extras["x_" + k] = np.mean([e[k] for e in res["extras_te"]], 0)
        if model in NESY2:   # event thresholds chosen on the validation split only (§4.1)
            from src.data.concept_targets import load_targets
            from src.eval.events import select_thresholds
            thr, curves = select_thresholds(res["fval"], val, load_targets(),
                                            "q" if model in Q_REGION else "cov")
            extras["event_thr"] = np.array([thr[0], thr[1], thr[2]], np.float32)
            extras["event_thr_curves"] = json.dumps(curves)
            for k in ("x_f_prob", "x_f_tprob"):
                extras[k] = extras[k].astype(np.float16)
        if model in DETECTORS:   # v3 §4.1 cross-fitting: out-of-fold detector outputs for the outer-train records
            from sklearn.model_selection import KFold
            T_ = load_targets()
            cf_prob = np.zeros((len(tr), 3, 900), np.float16)
            cf_part, cf_thr = np.zeros(len(tr), np.int8), []
            for k, (ia, ib) in enumerate(KFold(3, shuffle=True, random_state=seed).split(tr)):
                rr = fit_predict_dl(model, cfg, X, y, extra, tr[ia], tr[ib], seed + 101 * (k + 1), device, 1)
                th, _ = select_thresholds(rr["fval"], rr["val_idx"], T_, "q")
                cf_prob[ib] = rr["extras_te"][0]["f_prob"].astype(np.float16)
                cf_part[ib] = k
                cf_thr.append([th[0], th[1], th[2]])
            extras.update(cf_records=recs[tr], cf_prob=cf_prob, cf_part=cf_part, cf_thr=np.array(cf_thr, np.float32))
        np.savez(out, records_te=recs[te], y_te=y[te], p_te=p_te, mem_te=res["mem_te"],
                 records_va=recs[val], y_va=y[val], p_va=p_va, mem_va=res["mem_va"],
                 unc_total_te=tot_te, unc_epi_te=tot_te - ale_te, unc_ale_te=ale_te,
                 unc_total_va=tot_va, unc_epi_va=tot_va - ale_va, unc_ale_va=ale_va,
                 thr_gmean=gmean_threshold(y[val], p_va), thr_spec90=spec_threshold(y[val], p_va, 0.90),
                 cfg=json.dumps(cfg), cfgs=json.dumps(cfgs), inner_scores=json.dumps(scores),
                 temps=json.dumps(res["temps"]), epochs=json.dumps(res["epochs"]),
                 seconds=time.time() - t0, rule_w=res.get("rule_w", np.zeros(0)), **extras)
        return str(out), f"ok {time.time() - t0:.0f}s"
    except Exception:
        return str(out), "ERROR\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", default="W60")
    ap.add_argument("--label", default="y_ph705")
    ap.add_argument("--splits", default="outer_splits")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", default="0,1,2,3,4")
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--fixed_cfg_from", default=None)
    a = ap.parse_args()
    (ROOT / "results" / "cv" / a.exp).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cv" / a.exp / "run_args.json").write_text(json.dumps(vars(a), indent=2))
    jobs = [(a.model, a.window, a.label, a.splits, a.exp, int(r), int(f), a.ensemble, "cuda", a.fixed_cfg_from)
            for r in a.repeats.split(",") for f in a.folds.split(",")]
    t0 = time.time()
    if a.workers <= 1:
        for j in jobs:
            print(*run_fold(j), flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
            for res in pool.imap_unordered(run_fold, jobs):
                print(*res, flush=True)
    print(f"done {a.exp} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
