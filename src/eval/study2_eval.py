"""Evaluation of the end-to-end model analysis, exactly as pre-specified in docs/analiz_plani_v2.md §6 (written before any
NeSy-v2 result was seen).

Run: .venv\\Scripts\\python -m src.eval.study2_eval
Outputs: results/study2/study2_results.json (+ csv tables). Record-level Romagnoli derivatives are not
written (CC BY-NC-ND).
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.concept_fidelity import detector, match  # noqa: E402
from src.eval.events import BIN, MIN_BINS, bins_to_events, expert_events, expert_events_typed, in_region  # noqa: E402
from src.models.nesy import BINARY, RULES  # noqa: E402
from src.models.nesy_v2 import EXPERT_BITS, EXPERT_IDX  # noqa: E402

OUT = ROOT / "results" / "study2"
N_BOOT, SEED = 2000, 20261123
LAB = {0: "ACC", 1: "DEC", 2: "UC"}
TYPES = ["E", "L", "V", "P"]
RULE_TYPE = {"early": "E", "late": "L", "variable": "V", "prolonged": "P"}
VAL_FRAC, SEED_CV = 0.15, 20261122


# ----------------------------------------------------------------------------- loading
def load_nesy2(exp):
    files = sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz")))
    assert len(files) == 50, f"{exp}: {len(files)} folds"
    rows, frame, tprob, cbits, thr, rule_w, per_rep = [], {}, {}, {}, [], [], {}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        r = int(Path(fp).stem[1:].split("_f")[0])
        thr.append(z["event_thr"])
        rule_w.append(z["rule_w"].mean(0))
        for i, rec in enumerate(z["records_te"]):
            rec = str(rec)
            rows.append(dict(record=rec, repeat=r, y=int(z["y_te"][i]), p=float(z["p_te"][i])))
            frame.setdefault(rec, []).append(z["x_f_prob"][i].astype(np.float32))
            tprob.setdefault(rec, []).append(z["x_f_tprob"][i].astype(np.float32))
            cbits.setdefault(rec, []).append(z["x_c_bits"][i])
            per_rep[(rec, r)] = (z["x_f_prob"][i].astype(np.float32), z["event_thr"])
    df = pd.DataFrame(rows)
    return dict(df=df, frame={k: np.mean(v, 0) for k, v in frame.items()},
                tprob={k: np.mean(v, 0) for k, v in tprob.items()},
                cbits=pd.DataFrame({k: np.mean(v, 0) for k, v in cbits.items()}, index=BINARY).T,
                thr=np.median(np.array(thr), 0), thr_all=np.array(thr), rule_w=np.array(rule_w), per_rep=per_rep)


def repeat_avg(exp):
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        for i, rec in enumerate(z["records_te"]):
            rows.append(dict(record=str(rec), y=int(z["y_te"][i]), p=float(z["p_te"][i])))
    return pd.DataFrame(rows).groupby("record").agg(y=("y", "first"), p=("p", "mean"))


def head_bits(exp):
    acc = {}
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        for r, v in zip(z["records_te"], z["x_c_bits"]):
            acc.setdefault(str(r), []).append(v)
    return pd.DataFrame({k: np.mean(v, 0) for k, v in acc.items()}, index=BINARY).T


# ----------------------------------------------------------------------------- statistics
def pooled_f1(c):
    tp, nr, nd = c.sum(0)
    return 2 * tp / (nr + nd) if (nr + nd) else np.nan


def prf(c):
    tp, nr, nd = c.sum(0)
    return dict(se=tp / nr if nr else np.nan, ppv=tp / nd if nd else np.nan,
                f1=2 * tp / (nr + nd) if (nr + nd) else np.nan, n_ref=int(nr), n_det=int(nd), tp=int(tp))


def boot_f1_diff(ca, cb, rng):
    """ca, cb: (n_records, 3) counts [tp, n_ref, n_det]; paired record bootstrap of F1(a) - F1(b)."""
    obs = pooled_f1(ca) - pooled_f1(cb)
    n = len(ca)
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d.append(pooled_f1(ca[idx]) - pooled_f1(cb[idx]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=float(obs), lo=float(np.nanquantile(d, .025)), hi=float(np.nanquantile(d, .975)), p=float(min(p, 1)))


def boot_ci_f1(c, rng):
    n = len(c)
    v = [pooled_f1(c[rng.integers(0, n, n)]) for _ in range(N_BOOT)]
    return [float(pooled_f1(c)), float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]


def harrell_c(risk, ph):
    dr = risk[:, None] - risk[None, :]
    dp = ph[None, :] - ph[:, None]
    comp = dp > 0
    return ((dr > 0) & comp).sum() / comp.sum() + 0.5 * ((dr == 0) & comp).sum() / comp.sum()


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    T = load_targets()
    recs = [str(r) for r in T["records"]]
    pos = {r: i for i, r in enumerate(recs)}
    excl = set(json.loads((ROOT / "results" / "preprocess" / "exclusions.json").read_text())["excluded"])
    z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
    nz = load_nesy2("nesy2_W60")
    has_ablation = len(glob.glob(str(ROOT / "results" / "cv" / "nesy2nomask_W60" / "r*_f*.npz"))) == 50
    nzm = load_nesy2("nesy2nomask_W60") if has_ablation else None
    test_recs = sorted(set(nz["df"].record))
    assert not (set(test_recs) & excl) and len(test_recs) == 536
    res = {"n_records": len(test_recs), "event_thresholds_median": dict(zip(["ACC", "DEC", "UC"], nz["thr"].tolist()))}

    # ---------------- event-level counts: NeSy-v2, ablation, rule-based (§6.1)
    def model_events(frame, i, lab, thr):
        return bins_to_events(frame[lab], thr[lab], MIN_BINS[lab])

    rule_ev = {}
    for r in test_recs:
        i = pos[r]
        _, accs, decs, cons = detector(z["fhr"][i], z["fhr_mask"][i], z["uc"][i], z["uc_mask"][i])
        rule_ev[r] = {0: [(d["start"], d["end"]) for d in accs], 1: [(d["start"], d["end"]) for d in decs],
                      2: [(c["start"], c["end"]) for c in cons], "dtype": [RULE_TYPE.get(d["type"], "other") for d in decs]}

    def counts(get_det, region_key, iou=0.0):
        out = {}
        for lab in (0, 1, 2):
            c = []
            for r in test_recs:
                i = pos[r]
                reg = T[region_key][i].astype(float)
                ref, det = in_region(expert_events(T, i, lab), reg), in_region(get_det(r, lab), reg)
                c.append([len(match(ref, det, iou)), len(ref), len(det)])
            out[lab] = np.array(c)
        return out

    get_nz = lambda r, lab: model_events(nz["frame"][r], pos[r], lab, nz["thr"])  # noqa: E731
    get_rule = lambda r, lab: rule_ev[r][lab]  # noqa: E731
    C = {"nesy2": counts(get_nz, "q"), "rule": counts(get_rule, "q"),
         "nesy2_iou50": counts(get_nz, "q", 0.5), "rule_iou50": counts(get_rule, "q", 0.5),
         "nesy2_cov": counts(get_nz, "cov"), "rule_cov": counts(get_rule, "cov")}
    if nzm is not None:
        get_nzm = lambda r, lab: model_events(nzm["frame"][r], pos[r], lab, nzm["thr"])  # noqa: E731
        C["nomask"] = counts(get_nzm, "q")
    ev_tab = {}
    for k, cc in C.items():
        for lab in (0, 1, 2):
            ev_tab[f"{k}|{LAB[lab]}"] = {**prf(cc[lab]), "f1_ci": boot_ci_f1(cc[lab], rng)}
    res["events"] = ev_tab

    # ---------------- H1a/H1b
    h1 = {"H1a_DEC": boot_f1_diff(C["nesy2"][1], C["rule"][1], rng),
          "H1b_ACC": boot_f1_diff(C["nesy2"][0], C["rule"][0], rng)}
    # ---------------- H1c: accel window concept AUROC vs the weak-label NeSy head
    ok = [r for r in test_recs if T["annotated"][pos[r]] and T["rec_quality"][pos[r]]]
    y_acc = np.array([T["win_masked"][pos[r], EXPERT_BITS.index("accel")] for r in ok])
    assert (y_acc >= 0).all()
    v1 = head_bits("nesy_ens_W60")
    pa, pb = nz["cbits"].loc[ok, "accel"].values, v1.loc[ok, "accel"].values
    d, se, zz, p = M.delong_test(y_acc, pa, pb)
    h1["H1c_accel_window"] = dict(auroc_nesy2=float(roc_auc_score(y_acc, pa)), auroc_nesy1=float(roc_auc_score(y_acc, pb)),
                                  diff=d, lo=d - 1.96 * se, hi=d + 1.96 * se, p=p, n=len(ok), n_pos=int(y_acc.sum()))
    adj = M.holm([h1[k]["p"] for k in ("H1a_DEC", "H1b_ACC", "H1c_accel_window")])
    for k, a in zip(("H1a_DEC", "H1b_ACC", "H1c_accel_window"), adj):
        h1[k]["p_holm"] = float(a)
        h1[k]["supported"] = bool(h1[k]["diff"] > 0 and a < 0.05)
    res["H1"] = h1

    # per-repeat event F1 (each fold's own threshold)
    per_rep = {}
    for rep in range(5):
        f1s = {}
        for lab in (0, 1):
            c = []
            for r in test_recs:
                fr, th = nz["per_rep"][(r, rep)]
                i = pos[r]
                reg = T["q"][i].astype(float)
                ref = in_region(expert_events(T, i, lab), reg)
                det = in_region(bins_to_events(fr[lab], th[lab], MIN_BINS[lab]), reg)
                c.append([len(match(ref, det)), len(ref), len(det)])
            f1s[LAB[lab]] = float(pooled_f1(np.array(c)))
        per_rep[rep] = f1s
    res["per_repeat_event_f1"] = per_rep

    # ---------------- DEC type (matched pairs, region q)
    def type_pairs(kind):
        pairs = []
        for r in test_recs:
            i = pos[r]
            reg = T["q"][i].astype(float)
            ref_t = [(s, e, TYPES[t] if t >= 0 else "?") for s, e, t in expert_events_typed(T, i)]
            ref_t = [x for x in ref_t if in_region([(x[0], x[1])], reg)]
            if kind == "nesy2":
                det = bins_to_events(nz["frame"][r][1], nz["thr"][1], MIN_BINS[1])
                tp = nz["tprob"][r]
                det_t = [(s, e, TYPES[int(np.argmax(tp[:, s // BIN:max(s // BIN + 1, e // BIN)].mean(1)))]) for s, e in det]
            else:
                det_t = [(s, e, t) for (s, e), t in zip(rule_ev[r][1], rule_ev[r]["dtype"])]
            det_t = [x for x in det_t if in_region([(x[0], x[1])], reg)]
            for a, b in match([(s, e) for s, e, _ in ref_t], [(s, e) for s, e, _ in det_t]):
                pairs.append((ref_t[a][2], det_t[b][2]))
        return pairs
    dt = {}
    for kind in ("nesy2", "rule"):
        pr = [p for p in type_pairs(kind) if p[0] in TYPES]
        ye, yd = [p[0] for p in pr], [p[1] for p in pr]
        dt[kind] = dict(n_pairs=len(pr), macro_f1=float(f1_score(ye, yd, labels=TYPES, average="macro", zero_division=0)),
                        per_type_recall={t: float(np.mean([b == t for a, b in pr if a == t])) if any(a == t for a, _ in pr) else None for t in TYPES},
                        confusion=pd.crosstab(pd.Series(ye, name="expert"), pd.Series(yd, name="pred")).to_dict())
    res["dec_type"] = dt

    # ---------------- window concepts (§6.1)
    nocs = head_bits("nesy_nocsup_W60")
    weak = pd.read_csv(ROOT / "results" / "features" / "figo_W60.csv", dtype={"record": str}).set_index("record")
    wc = {}
    for j, b in enumerate(EXPERT_BITS):
        yy = np.array([T["win_masked"][pos[r], j] for r in ok])
        if (yy == 1).sum() < 20:
            wc[b] = {"reported": False, "n_pos": int((yy == 1).sum())}
            continue
        row = {"reported": True, "n_pos": int((yy == 1).sum()), "n": len(ok)}
        for name, P in (("nesy2", nz["cbits"]), ("nesy1", v1), ("nesy1_nocsup", nocs)):
            pp = P.loc[ok, b].values
            row[f"auroc_{name}"] = float(roc_auc_score(yy, pp))
            row[f"kappa_{name}"] = float(cohen_kappa_score(yy, (pp >= 0.5).astype(int)))
        wl = weak.loc[ok, f"c_{b}"].astype(int).values
        row["kappa_weak"] = float(cohen_kappa_score(yy, wl))
        wc[b] = row
    res["window_concepts"] = wc

    # ---------------- H2 rule weights
    rw = nz["rule_w"]               # (50, 10) fold-level member means
    ip, isus = RULES.index("pathological"), RULES.index("suspicious")
    med = float(np.median(rw[:, ip]))
    frac = float((rw[:, ip] > rw[:, isus]).mean())
    res["H2"] = dict(rule_weights_median={k: float(np.median(rw[:, i])) for i, k in enumerate(RULES)},
                     rule_weights_p2_5={k: float(np.quantile(rw[:, i], .025)) for i, k in enumerate(RULES)},
                     rule_weights_p97_5={k: float(np.quantile(rw[:, i], .975)) for i, k in enumerate(RULES)},
                     path_median=med, frac_path_gt_susp=frac,
                     supported=bool(abs(med - 0.10) >= 0.05 and frac >= 0.90))

    # ---------------- H3 / H4 and secondary discrimination & calibration
    # continuous pH: evaluation reference only (never an input)
    meta = pd.read_csv(ROOT / "results" / "eda" / "metadata_all.csv", dtype={"record": str}).set_index("record")
    ph_col = "ph" if "ph" in meta.columns else None
    a2 = nz["df"].groupby("record").agg(y=("y", "first"), p=("p", "mean")).loc[test_recs]
    cb = repeat_avg("cnn_ens_W60").loc[test_recs]
    y = a2.y.values
    d3 = M.paired_bootstrap_diff(y, a2.p.values, cb.p.values, M.auroc, n=N_BOOT, seed=SEED)
    dl = M.delong_test(y, a2.p.values, cb.p.values)
    rep_auc = nz["df"].groupby("repeat").apply(lambda g: roc_auc_score(g.y, g.p))
    res["H3"] = dict(auroc_nesy2=M.bootstrap_ci(y, a2.p.values, M.auroc, n=N_BOOT, seed=SEED),
                     auroc_cnn_bayes=M.bootstrap_ci(y, cb.p.values, M.auroc, n=N_BOOT, seed=SEED),
                     diff=d3[0], lo=d3[1], hi=d3[2], p_boot=d3[3], delong_p=dl[3],
                     noninferior=bool(d3[1] > -0.05), auroc_per_repeat_mean=float(rep_auc.mean()),
                     auroc_per_repeat_sd=float(rep_auc.std()))
    sec = {}
    for name, fn in (("auprc", M.auprc), ("brier", M.brier), ("ece", M.ece)):
        sec[name] = dict(nesy2=M.bootstrap_ci(y, a2.p.values, fn, n=N_BOOT, seed=SEED),
                         cnn_bayes=M.bootstrap_ci(y, cb.p.values, fn, n=N_BOOT, seed=SEED),
                         diff=M.paired_bootstrap_diff(y, a2.p.values, cb.p.values, fn, n=N_BOOT, seed=SEED))
    if nzm is not None:
        am = nzm["df"].groupby("record").agg(p=("p", "mean")).loc[test_recs]
        sec["nomask_auroc_diff_vs_nesy2"] = M.paired_bootstrap_diff(y, am.p.values, a2.p.values, M.auroc, n=N_BOOT, seed=SEED)
    res["discrimination_calibration"] = sec
    if ph_col:
        ph = meta.loc[test_recs, ph_col].astype(float).values
        okp = np.isfinite(ph)
        r2, r1, phh = a2.p.values[okp], cb.p.values[okp], ph[okp]
        c2, c1 = harrell_c(r2, phh), harrell_c(r1, phh)
        bs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(phh), len(phh))
            bs.append(harrell_c(r2[idx], phh[idx]) - harrell_c(r1[idx], phh[idx]))
        res["H4"] = dict(c_nesy2=float(c2), c_cnn_bayes=float(c1), diff=float(c2 - c1),
                         lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(okp.sum()))
    else:
        res["H4"] = "pH column not found in results/eda/labels.csv"

    # ---------------- leakage: concept intervention (§6.2), approximate (member-averaged logits)
    from src.models.nesy import FIGOLogic
    import torch
    splits = json.loads((ROOT / "results" / "splits" / "outer_splits.json").read_text())
    lab_all = pd.read_csv(ROOT / "results" / "eda" / "labels.csv", dtype={"record": str}).set_index("record")["y_ph705"]

    def intervention(exp):
        rows = []
        for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
            zz = np.load(fp, allow_pickle=True)
            rr, ff = map(int, Path(fp).stem[1:].split("_f"))
            tr = np.array(splits["repeats"][rr][ff]["train"])
            ytr = lab_all.reindex(tr).fillna(0).astype(int).values
            seed = SEED_CV + 1000 * rr + ff
            fit, _ = train_test_split(np.arange(len(tr)), test_size=VAL_FRAC, stratify=ytr, random_state=seed)
            npos = max(ytr[fit].sum(), 1)
            shift = math.log((len(fit) - npos) / npos)
            Tm = float(np.mean(json.loads(str(zz["temps"]))))
            logic = FIGOLogic()
            w = np.clip(zz["rule_w"].mean(0), 1e-6, 1 - 1e-6)
            logic.rule_logit.data = torch.tensor(np.log(w / (1 - w)), dtype=torch.float32)
            lb, lv, lbits = (torch.tensor(zz[k], dtype=torch.float32) for k in ("x_lb", "x_lv", "x_lbits"))
            sel = [k for k, rec in enumerate(zz["records_te"]) if T["annotated"][pos[str(rec)]] and T["rec_quality"][pos[str(rec)]]]
            if not sel:
                continue
            ex = torch.tensor(np.array([T["win_masked"][pos[str(zz["records_te"][k])]] for k in sel]), dtype=torch.float32)
            li = lbits[sel].clone()
            li[:, EXPERT_IDX] = torch.where(ex > 0.5, torch.tensor(9.2), torch.tensor(-9.2))
            with torch.no_grad():
                l0, _ = logic(lb[sel], lv[sel], lbits[sel])
                l1, _ = logic(lb[sel], lv[sel], li)
            sig = lambda t: 1 / (1 + np.exp(-((t.numpy() - shift) / Tm)))  # noqa: E731
            for k, p0, p1 in zip(sel, sig(l0), sig(l1)):
                rows.append(dict(record=str(zz["records_te"][k]), y=int(zz["y_te"][k]), p0=float(p0), p1=float(p1)))
        d = pd.DataFrame(rows).groupby("record").agg(y=("y", "first"), p0=("p0", "mean"), p1=("p1", "mean"))
        ll = lambda yy, p: float(-np.mean(yy * np.log(np.clip(p, 1e-7, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-7, 1))))  # noqa: E731
        return dict(n=len(d), n_pos=int(d.y.sum()), auroc_model=float(roc_auc_score(d.y, d.p0)),
                    auroc_intervened=float(roc_auc_score(d.y, d.p1)), logloss_model=ll(d.y.values, d.p0.values),
                    logloss_intervened=ll(d.y.values, d.p1.values), mean_abs_dp=float(np.mean(np.abs(d.p1 - d.p0))))
    res["leakage_intervention"] = {"nesy2_W60": intervention("nesy2_W60"), "nesy_ens_W60": intervention("nesy_ens_W60"),
                                   "note": "member-averaged concept logits and rule weights; approximate, see plan §6.2"}

    # ---------------- K3 reference-standard audit (§6.4)
    wq = pd.read_csv(ROOT / "results" / "preprocess" / "window_quality.csv", dtype={"record": str})
    wq = wq[wq.window == "W60"].set_index("record")
    ann = [r for r in test_recs if T["annotated"][pos[r]]]
    dens = []
    for r in ann:
        i = pos[r]
        cov_h = T["cov"][i].sum() * 4 / 3600
        n_ev = int(((T["ev_rec"] == i) & (T["ev_label"] <= 1)).sum())       # ACC + DEC
        dens.append(dict(record=r, density=n_ev / max(cov_h, 1e-6), miss=float(wq.at[r, "fhr_missing"]),
                         y=int(lab_all.get(r, 0)), masked_frac=1 - float(T["q"][i].sum() / max(T["cov"][i].sum(), 1))))
    dd = pd.DataFrame(dens)
    dd["tert"] = pd.qcut(dd.miss, 3, labels=["low", "mid", "high"])
    k3 = dict(spearman_density_vs_missing=float(spearmanr(dd.density, dd.miss).statistic), by_tertile={})
    for t, g in dd.groupby("tert", observed=True):
        a, b = g[g.y == 1].density, g[g.y == 0].density
        k3["by_tertile"][t] = dict(n=len(g), n_pos=int(len(a)), med_density_pos=float(a.median()) if len(a) else None,
                                   med_density_neg=float(b.median()),
                                   mannwhitney_p=float(mannwhitneyu(a, b).pvalue) if len(a) >= 3 else None)
    k3["masked_frac_mean_by_outcome"] = dd.groupby("y").masked_frac.mean().to_dict()
    res["K3_reference_audit"] = k3

    (OUT / "study2_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("H1", "H2", "H3", "H4")}, indent=1, default=float))


if __name__ == "__main__":
    main()
