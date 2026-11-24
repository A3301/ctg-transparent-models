"""Plan B / Plan C evaluation, as pre-specified in docs/analiz_plani_v3.md §5-6.
Written before any v3 result was seen (only "ok" lines of a test fold were seen).

Run: .venv\\Scripts\\python -m src.eval.study3_eval [--skip_b]
Output: results/study3/study3_results.json (aggregate numbers only; no record-level Romagnoli derivatives).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.concept_fidelity import detector, match  # noqa: E402
from src.eval.events import MIN_BINS, bins_to_events, expert_events, in_region  # noqa: E402
from src.eval.study2_eval import (N_BOOT, SEED, boot_ci_f1, boot_f1_diff, harrell_c, load_nesy2,  # noqa: E402
                                  prf, repeat_avg)

OUT = ROOT / "results" / "study3"
LAB = {0: "ACC", 1: "DEC", 2: "UC"}
NI_MARGIN = -0.05
MODELS_C = ["transp_gam", "transp_lr", "transp_gam_noq", "transp_gam_nohrv"]
REFS = {"cnn_bayes": "cnn_ens_W60", "figo_lr": "figo_lr_W60", "georgoulas": "georgoulas_W60"}


def n_folds(exp):
    return len(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz")))


def discrimination(preds, y, recs, ph):
    out = {}
    for k, p in preds.items():
        out[k] = dict(auroc=M.bootstrap_ci(y, p, M.auroc, n=N_BOOT, seed=SEED),
                      auprc=M.bootstrap_ci(y, p, M.auprc, n=N_BOOT, seed=SEED),
                      brier=M.bootstrap_ci(y, p, M.brier, n=N_BOOT, seed=SEED),
                      ece=M.bootstrap_ci(y, p, M.ece, n=N_BOOT, seed=SEED))
        ok = np.isfinite(ph)
        out[k]["harrell_c"] = float(harrell_c(p[ok], ph[ok]))
    return out


def per_repeat_auroc(exp):
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        r = int(Path(fp).stem[1:].split("_f")[0])
        rows += [dict(repeat=r, y=int(a), p=float(b)) for a, b in zip(z["y_te"], z["p_te"])]
    d = pd.DataFrame(rows)
    v = d.groupby("repeat").apply(lambda g: M.auroc(g.y.values, g.p.values))
    return dict(mean=float(v.mean()), sd=float(v.std()))


def shape_functions(recs_all, y_all):
    """§6.2: per-feature log-odds contribution of transp_gam, repeat 0 (10 outer folds), refit with the
    saved per-fold config and seed (deterministic), other features at their median."""
    from src.run.cv import SEED as CV_SEED, VAL_FRAC, transp_table
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    t = transp_table("transp_gam", "W60")
    X, cols = t.values.astype(float), list(t.columns)
    pos = {k: i for i, k in enumerate(t.index.astype(str))}
    y = np.nan_to_num(y_all.reindex(t.index.astype(str)).values, nan=0).astype(int)
    splits = json.loads((ROOT / "results" / "splits" / "outer_splits.json").read_text())
    curves = {c: [] for c in cols}
    for f in range(10):
        z = np.load(ROOT / "results" / "cv" / "transp_gam_W60" / f"r0_f{f}.npz", allow_pickle=True)
        cfg = json.loads(str(z["cfg"]))
        tr = np.array([pos[k] for k in splits["repeats"][0][f]["train"]])
        fit, _ = train_test_split(tr, test_size=VAL_FRAC, stratify=y[tr], random_state=CV_SEED + f)
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            SplineTransformer(n_knots=cfg["n_knots"], degree=3, knots="quantile", extrapolation="linear"),
                            LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000)).fit(X[fit], y[fit])
        med = np.nanmedian(X[fit], 0)
        for j, c in enumerate(cols):
            lo, hi = np.nanpercentile(X[fit, j], [5, 95])
            grid = np.linspace(lo, hi, 25)
            G = np.tile(med, (len(grid), 1))
            G[:, j] = grid
            lp = clf.decision_function(G)
            curves[c].append((grid, lp - lp[len(grid) // 2]))
    summ = {}
    for c, cs in curves.items():
        rng_ = [float(np.ptp(v)) for _, v in cs]
        summ[c] = dict(range_median=float(np.median(rng_)), range_min=float(np.min(rng_)), range_max=float(np.max(rng_)),
                       direction=float(np.median([np.sign(v[-1] - v[0]) for _, v in cs])))
    top = sorted(summ, key=lambda c: -summ[c]["range_median"])[:8]
    return {"top8": top, "all": summ}, curves


def detector_fidelity(T, test_recs, pos, z, rng, exp="det_W60"):
    nz = load_nesy2(exp)
    assert set(nz["df"].record) == set(test_recs)
    rule_ev = {}
    for r in test_recs:
        i = pos[r]
        _, accs, decs, cons = detector(z["fhr"][i], z["fhr_mask"][i], z["uc"][i], z["uc_mask"][i])
        rule_ev[r] = {0: [(d["start"], d["end"]) for d in accs], 1: [(d["start"], d["end"]) for d in decs],
                      2: [(c["start"], c["end"]) for c in cons]}

    def counts(get_det, iou=0.0):
        out = {}
        for lab in (0, 1, 2):
            c = []
            for r in test_recs:
                i = pos[r]
                reg = T["q"][i].astype(float)
                ref, det = in_region(expert_events(T, i, lab), reg), in_region(get_det(r, lab), reg)
                c.append([len(match(ref, det, iou)), len(ref), len(det)])
            out[lab] = np.array(c)
        return out
    get_det = lambda r, lab: bins_to_events(nz["frame"][r][lab], nz["thr"][lab], MIN_BINS[lab])  # noqa: E731
    get_rule = lambda r, lab: rule_ev[r][lab]  # noqa: E731
    C = {"det": counts(get_det), "rule": counts(get_rule), "det_iou50": counts(get_det, 0.5),
         "rule_iou50": counts(get_rule, 0.5)}
    ev = {f"{k}|{LAB[lab]}": {**prf(cc[lab]), "f1_ci": boot_ci_f1(cc[lab], rng)} for k, cc in C.items() for lab in (0, 1, 2)}
    b1 = {"B1a_DEC": boot_f1_diff(C["det"][1], C["rule"][1], rng), "B1b_ACC": boot_f1_diff(C["det"][0], C["rule"][0], rng)}
    return ev, b1, nz["thr"]


def fhrma_detector(thr, exp="det_W60"):
    """§6.3: repeat-0 detector ensemble (50 members) on FHRMA, v2 §6.5 method; 5th channel = baseline deviation."""
    from src.data import annotations as A
    from src.data.inputs import FHR_C, FHR_S, UC_C, UC_S, _down
    from src.eval.concept_fidelity import _intervals
    from src.features import figo
    from src.models.det import DetNet
    from src.preprocess.clean import clean_fhr, clean_uc
    nets = []
    rule_ch = exp.startswith("detk")
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "weights" / "r0_f*_m*.pt"))):
        ck = torch.load(fp, map_location="cuda", weights_only=False)
        net = DetNet(8 if rule_ch else 5, **{k: v for k, v in ck["cfg"].items() if k in ("width", "n_stages", "kernel", "drop")}).cuda()
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append(net)
    assert len(nets) == 50, len(nets)
    WIN, BIN = 14400, 16
    E = {k: v for k, v in A.fhrma_expert().items() if k.startswith("train")}
    counts = {"det": {0: [], 1: []}, "rule": {0: [], 1: []}}
    for name in sorted(E):
        v, sig = E[name], A.fhrma_signal(name)
        n = len(sig["fhr1"])
        cf = clean_fhr(sig["fhr1"])
        fhr, fm = np.nan_to_num(cf["fhr"], nan=0.0).astype(np.float32), cf["mask"].astype(np.float32)
        if np.all(sig["toco"] == 0):
            uc, um = np.zeros(n, np.float32), np.zeros(n, np.float32)
        else:
            cu = clean_uc(sig["toco"])
            uc, um = np.nan_to_num(cu["uc"], nan=0.0).astype(np.float32), cu["mask"].astype(np.float32)
        valid = fm > 0.5
        base = figo.baseline_curve(fhr, valid)
        dv_ok = valid & np.isfinite(base)
        dev, dm = np.where(dv_ok, fhr - base, 0.0).astype(np.float32), dv_ok.astype(np.float32)
        nw = int(np.ceil(n / WIN))
        pad = nw * WIN - n
        f_, fm_, u_, um_, d_, dm_ = (np.concatenate([a, np.zeros(pad, np.float32)]).reshape(nw, WIN)
                                     for a in (fhr, fm, uc, um, dev, dm))
        f1, m1 = _down(f_, fm_, 4)
        u1, n1 = _down(u_, um_, 4)
        d1, _ = _down(d_, dm_, 4)
        Xw = np.stack([np.where(m1 > 0, (f1 - FHR_C) / FHR_S, 0), m1, np.where(n1 > 0, (u1 - UC_C) / UC_S, 0), n1,
                       d1 / 25.0], 1)
        if rule_ch:   # ek2: rule-event indicators computed on the whole record, then cut into the same windows
            from src.run.cv import rule_event_indicators
            pad4 = np.concatenate([fhr, np.zeros(pad, np.float32)]), np.concatenate([fm, np.zeros(pad, np.float32)]),                 np.concatenate([uc, np.zeros(pad, np.float32)]), np.concatenate([um, np.zeros(pad, np.float32)])
            rc = rule_event_indicators(*pad4).reshape(3, nw, WIN // 4).transpose(1, 0, 2)
            Xw = np.concatenate([Xw, rc], 1)
        X = torch.tensor(Xw, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            P = np.mean([net(X)["f_prob"].cpu().numpy() for net in nets], 0)
        P = np.concatenate(list(P), axis=-1)[:, : int(np.ceil(n / BIN))]
        nb = P.shape[1]
        excl = np.zeros(nb * BIN, bool)
        for s, e in _intervals(v["notToAnalyse"], n) + _intervals(v["unreliableSignal"], n):
            excl[s:e] = True
        region = ((~excl[: nb * BIN].reshape(nb, BIN)).mean(1) >= 0.5) & ((np.arange(nb) * BIN) < n)
        rule = {0: figo.detect_events(fhr, valid, base, +1), 1: figo.detect_events(fhr, valid, base, -1)}
        for lab, field in ((0, "accelerations"), (1, "decelerations")):
            ref = in_region([(s, e) for s, e in _intervals(v[field], n) if e > s], region.astype(float))
            for kind, det in (("det", bins_to_events(P[lab], thr[lab], MIN_BINS[lab])),
                              ("rule", [(d["start"], d["end"]) for d in rule[lab]])):
                det = in_region(det, region.astype(float))
                counts[kind][lab].append([len(match(ref, det)), len(ref), len(det)])
    rng = np.random.default_rng(SEED)
    out = {"n_records": len(E)}
    for lab in (0, 1):
        a, b = np.array(counts["det"][lab]), np.array(counts["rule"][lab])
        out[LAB[lab]] = {"det": prf(a), "rule": prf(b), **boot_f1_diff(a, b, rng)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_b", action="store_true", help="evaluate Plan C only (Plan B not finished)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    from src.data.inputs import labels
    excl = set(json.loads((ROOT / "results" / "preprocess" / "exclusions.json").read_text())["excluded"])
    for m in MODELS_C:
        assert n_folds(f"{m}_W60") == 50, m
    base = repeat_avg("cnn_ens_W60")
    test_recs = sorted(base.index)
    assert len(test_recs) == 536 and not (set(test_recs) & excl)
    y = base.loc[test_recs].y.values
    meta = pd.read_csv(ROOT / "results" / "eda" / "metadata_all.csv", dtype={"record": str}).set_index("record")
    ph = meta.loc[test_recs, "ph"].astype(float).values
    preds = {k: repeat_avg(v).loc[test_recs].p.values for k, v in REFS.items()}
    for m in MODELS_C:
        preds[m] = repeat_avg(f"{m}_W60").loc[test_recs].p.values
    have_b = (not a.skip_b) and n_folds("transpdet_gam_W60") == 50 and n_folds("det_W60") == 50
    if have_b:
        preds["transpdet_gam"] = repeat_avg("transpdet_gam_W60").loc[test_recs].p.values
    for k, p in preds.items():
        assert repeat_avg(REFS.get(k, f"{k}_W60")).loc[test_recs].y.values.tolist() == y.tolist(), k
    res = {"n_records": len(test_recs), "n_pos": int(y.sum()), "plan_b_included": have_b}

    # ---- non-inferiority family (C1, B2): two-sided 97.5% CI lower bound > -0.05
    ni = {}
    for code, m in (("C1", "transp_gam"), ("B2", "transpdet_gam")):
        if m not in preds:
            continue
        d = M.paired_bootstrap_diff(y, preds[m], preds["cnn_bayes"], M.auroc, n=N_BOOT, seed=SEED, alpha=0.025)
        ni[code] = dict(model=m, diff=d[0], lo_97_5=d[1], hi_97_5=d[2], p_boot=d[3],
                        delong_p=M.delong_test(y, preds[m], preds["cnn_bayes"])[3], noninferior=bool(d[1] > NI_MARGIN))
    res["noninferiority"] = ni

    # ---- superiority family (C2, B1a, B1b): Holm
    sup = {}
    d = M.paired_bootstrap_diff(y, preds["transp_gam"], preds["figo_lr"], M.auroc, n=N_BOOT, seed=SEED)
    sup["C2"] = dict(diff=d[0], lo=d[1], hi=d[2], p=d[3], delong_p=M.delong_test(y, preds["transp_gam"], preds["figo_lr"])[3])
    if have_b:
        T = load_targets()
        pos = {r: i for i, r in enumerate(str(x) for x in T["records"])}
        z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
        ev, b1, thr = detector_fidelity(T, test_recs, pos, z, rng)
        res["detector_events"], res["detector_thresholds_median"] = ev, thr.tolist()
        sup.update(b1)
        try:
            res["detector_fhrma"] = fhrma_detector(thr)
        except Exception as e:  # noqa: BLE001 - secondary analysis; report, do not hide
            res["detector_fhrma"] = f"failed: {e!r}"
    keys = [k for k in ("C2", "B1a_DEC", "B1b_ACC") if k in sup]
    for k, adj in zip(keys, M.holm([sup[k]["p"] for k in keys])):
        sup[k]["p_holm"] = float(adj)
        sup[k]["supported"] = bool(sup[k]["diff"] > 0 and adj < 0.05)
    res["superiority"] = sup

    # ---- B3 (descriptive) and secondary analyses
    if have_b:
        d = M.paired_bootstrap_diff(y, preds["transpdet_gam"], preds["transp_gam"], M.auroc, n=N_BOOT, seed=SEED)
        res["B3_transpdet_minus_transp"] = dict(diff=d[0], lo=d[1], hi=d[2], p=d[3])
    res["discrimination"] = discrimination(preds, y, test_recs, ph)
    res["per_repeat_auroc"] = {k: per_repeat_auroc(REFS.get(k, f"{k}_W60")) for k in preds}
    avg = (preds["transp_gam"] + preds["cnn_bayes"]) / 2
    res["complementarity_avg_transp_cnn"] = dict(
        auroc=M.bootstrap_ci(y, avg, M.auroc, n=N_BOOT, seed=SEED),
        diff_vs_cnn=M.paired_bootstrap_diff(y, avg, preds["cnn_bayes"], M.auroc, n=N_BOOT, seed=SEED))
    res["informative_missingness"] = dict(
        transp_minus_noq=M.paired_bootstrap_diff(y, preds["transp_gam"], preds["transp_gam_noq"], M.auroc, n=N_BOOT, seed=SEED),
        transp_minus_nohrv=M.paired_bootstrap_diff(y, preds["transp_gam"], preds["transp_gam_nohrv"], M.auroc, n=N_BOOT, seed=SEED))
    shp, curves = shape_functions(test_recs, labels("y_ph705"))
    res["shape_functions"] = shp
    np.save(OUT / "shape_curves_r0.npy", np.array([(c, [(g.tolist(), v.tolist()) for g, v in cs]) for c, cs in curves.items()],
                                                   dtype=object), allow_pickle=True)
    (OUT / "study3_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("noninferiority", "superiority")}, indent=1, default=float))


if __name__ == "__main__":
    main()
