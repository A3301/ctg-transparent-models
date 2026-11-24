"""Fidelity of the rule-based FIGO extractor (and of the weak-label NeSy concept heads) against
expert annotations. Descriptive measurement only: no threshold of src/features/figo.py is
changed here, and no outcome variable is read.

Protocol (fixed before the first run, 2026-11-23; reported in docs/kavram_dogrulugu.md)
------------------------------------------------------------------------------------
CTU-UHB vs Romagnoli 2020
  * Unit: the W60 window (last 60 min), i.e. the window the benchmark weak labels came from;
    the 536 records included in the benchmark. Records whose Romagnoli file holds no event at all
    are treated as not annotated and excluded (count reported).
  * Expert coverage inside the window = window ∩ [0, annotation length). Expert events are
    clipped to it; detector events count if they overlap it.
  * Event matching (ACC, DEC, UC): one-to-one, greedy by largest overlap. Primary criterion:
    any temporal overlap; secondary: IoU >= 0.5. Pooled sensitivity, PPV, F1 with record-level
    bootstrap 95 % CIs (1000 resamples, seed 20261123). Per-window counts: Spearman rho.
  * Deceleration type: confusion of expert E/L/V/P vs detector type over matched pairs.
  * Window concepts: expert-derived concepts use the same definitions as
    figo.concepts_and_class, computed from expert events: accel (any ACC), early (any E),
    prolonged (any P), very_prolonged (any DEC > 300 s), rep_late / rep_variable / rep_any
    (> 50 % of expert contractions have a deceleration of that type whose FHR nadir lies within
    +/-60 s of the contraction's UC peak; nadir/peak taken from the cleaned signals inside the
    expert intervals). Agreement of weak labels: prevalence, Se, Sp, Cohen's kappa (bootstrap CI).
  * Weak-label NeSy concept heads (nesy_ens_W60; contrast nesy_nocsup_W60): out-of-fold test
    probabilities averaged over the 5 repeats; AUROC against expert-derived and weak labels.
FHRMA (66 training records, other centre)
  * Whole record; FHR1 cleaned with clean_fhr; samples in expert notToAnalyse / unreliableSignal
    are removed from every comparison. Baseline: MAE and RMSD (bpm) between the extractor's
    baseline curve and the expert baseline over valid samples. ACC/DEC: matching as above.

Run: .venv\\Scripts\\python -m src.eval.concept_fidelity
Outputs: results/annotations/concept_fidelity.json (+ csv tables).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import annotations as A  # noqa: E402
from src.features import figo  # noqa: E402
from src.models.nesy import BINARY  # noqa: E402
from src.preprocess.clean import clean_fhr  # noqa: E402

FS = 4
SEED = 20261123
N_BOOT = 1000
OUT = ROOT / "results" / "annotations"
TYPE_MAP = {"E": "early", "L": "late", "V": "variable", "P": "prolonged"}
CONCEPTS = ["accel", "early", "prolonged", "very_prolonged", "rep_late", "rep_variable", "rep_any"]


# ----------------------------------------------------------------------------- matching
def match(a: list[tuple[int, int]], b: list[tuple[int, int]], iou_min: float = 0.0):
    """Greedy one-to-one matching by overlap. Returns list of (i, j)."""
    pairs = []
    for i, (s1, e1) in enumerate(a):
        for j, (s2, e2) in enumerate(b):
            ov = min(e1, e2) - max(s1, s2)
            if ov <= 0:
                continue
            iou = ov / (max(e1, e2) - min(s1, s2))
            if iou >= iou_min:
                pairs.append((ov, i, j))
    used_a, used_b, out = set(), set(), []
    for ov, i, j in sorted(pairs, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); out.append((i, j))
    return out


def counts(a, b, iou_min):
    m = len(match(a, b, iou_min))
    return {"tp": m, "n_ref": len(a), "n_det": len(b)}


def prf(c):
    se = c["tp"] / c["n_ref"] if c["n_ref"] else np.nan
    ppv = c["tp"] / c["n_det"] if c["n_det"] else np.nan
    f1 = 2 * c["tp"] / (c["n_ref"] + c["n_det"]) if (c["n_ref"] + c["n_det"]) else np.nan
    return se, ppv, f1


def boot_prf(per_rec: pd.DataFrame, rng) -> dict:
    """per_rec: columns tp, n_ref, n_det (one row per record). Pooled metrics + bootstrap CI."""
    tot = per_rec[["tp", "n_ref", "n_det"]].sum()
    est = prf(tot)
    arr = per_rec[["tp", "n_ref", "n_det"]].to_numpy()
    bs = []
    for _ in range(N_BOOT):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        bs.append(prf({"tp": s[0], "n_ref": s[1], "n_det": s[2]}))
    bs = np.array(bs, float)
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    return {k: [round(float(est[i]), 4), round(float(lo[i]), 4), round(float(hi[i]), 4)]
            for i, k in enumerate(["sensitivity", "ppv", "f1"])} | {
        "n_ref": int(tot["n_ref"]), "n_det": int(tot["n_det"]), "tp": int(tot["tp"])}


# ----------------------------------------------------------------------------- detector
def detector(fhr, fhr_mask, uc, uc_mask):
    valid, uc_valid = fhr_mask > 0.5, uc_mask > 0.5
    base = figo.baseline_curve(fhr, valid)
    accs = figo.detect_events(fhr, valid, base, +1)
    decs = figo.detect_events(fhr, valid, base, -1)
    cons, _ = figo.detect_contractions(uc, uc_valid)
    decs = figo.classify_decels(decs, cons, uc_valid)
    return base, accs, decs, cons


def expert_concepts(decs, cons, fhr, valid, uc, uc_valid):
    """decs: list of (s, e, type); cons: list of (s, e). Same definitions as figo.concepts_and_class."""
    def nadir(s, e):
        seg = np.where(valid[s:e], fhr[s:e], np.inf)
        return s + int(np.argmin(seg)) if np.isfinite(seg).any() else (s + e) // 2

    def peak(s, e):
        seg = np.where(uc_valid[s:e], uc[s:e], -np.inf)
        return s + int(np.argmax(seg)) if np.isfinite(seg).any() else (s + e) // 2

    nad = [(nadir(s, e), t) for s, e, t in decs]
    pks = [peak(s, e) for s, e in cons]
    aw = figo.T["assoc_window_s"] * FS

    def frac(t):
        if not pks:
            return 0.0
        return np.mean([any((t is None or tt == t) and abs(n - p) <= aw for n, tt in nad) for p in pks])

    rf = figo.T["repetitive_frac"]
    return {"rep_late": int(frac("late") > rf), "rep_variable": int(frac("variable") > rf),
            "rep_any": int(frac(None) > rf),
            "early": int(any(t == "early" for *_, t in decs)),
            "prolonged": int(any(t == "prolonged" for *_, t in decs)),
            "very_prolonged": int(any((e - s) / FS > figo.T["very_prolonged_s"] for s, e, _ in decs))}


def one_record(rec, sig, ws, ann_len, ev):
    fhr, fm, uc, um = sig
    W = len(fhr)
    c0, c1 = max(ws, 0) - ws, min(ws + W, ann_len) - ws         # covered part, window coords
    if c1 <= c0:
        return None
    base, accs, decs, cons = detector(fhr, fm, uc, um)
    ov = lambda s, e: min(e, c1) - max(s, c0) > 0  # noqa: E731
    det = {"ACC": [(d["start"], d["end"]) for d in accs if ov(d["start"], d["end"])],
           "DEC": [(d["start"], d["end"]) for d in decs if ov(d["start"], d["end"])],
           "UC": [(c["start"], c["end"]) for c in cons if ov(c["start"], c["end"])]}
    det_dec_type = [d["type"] for d in decs if ov(d["start"], d["end"])]
    ex = {"ACC": [], "DEC": [], "UC": []}
    ex_dec_type = []
    for _, e in ev.iterrows():
        s, t = max(int(e.start) - ws, c0), min(int(e.end) - ws, c1)
        if t <= s or e.label not in ex:
            continue
        ex[e.label].append((s, t))
        if e.label == "DEC":
            ex_dec_type.append(TYPE_MAP.get(e.subtype, "unknown"))
    row = {"rec": rec, "covered_min": (c1 - c0) / FS / 60}
    for lab in ex:
        for tag, thr in (("any", 0.0), ("iou50", 0.5)):
            c = counts(ex[lab], det[lab], thr)
            row.update({f"{lab}_{tag}_{k}": v for k, v in c.items()})
    pairs = [(ex_dec_type[i], det_dec_type[j]) for i, j in match(ex["DEC"], det["DEC"], 0.0)]
    exc = expert_concepts([(s, e, t) for (s, e), t in zip(ex["DEC"], ex_dec_type)], ex["UC"],
                          fhr, fm > 0.5, uc, um > 0.5)
    exc["accel"] = int(len(ex["ACC"]) > 0)
    row.update({f"exp_{k}": v for k, v in exc.items()})
    return row, pairs


# ----------------------------------------------------------------------------- CTU-UHB
def ctu_uhb(rng) -> dict:
    z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
    recs = [str(r) for r in z["records"]]
    wq = pd.read_csv(ROOT / "results" / "preprocess" / "window_quality.csv", dtype={"record": str})
    ws = wq[wq.window == "W60"].set_index("record")["start"].to_dict()
    excl = set(json.loads((ROOT / "results" / "preprocess" / "exclusions.json").read_text())["excluded"])
    ev = pd.read_csv(ROOT / "data" / "processed" / "romagnoli_events.csv", dtype={"subtype": str})
    ev["rec"] = ev["rec"].astype(str)
    summ = json.loads((OUT / "romagnoli_summary.json").read_text())
    ann_len = {str(k): v for k, v in summ["ann_len_by_record"].items()}
    has_ev = set(ev.rec)
    use = [r for r in recs if r not in excl and r in has_ev]
    info = {"n_records_w60": len(recs), "excluded_study1": len(excl),
            "excluded_no_expert_events": sum(1 for r in recs if r not in excl and r not in has_ev)}
    arr = {k: z[k] for k in ("fhr", "fhr_mask", "uc", "uc_mask")}
    pos = {r: i for i, r in enumerate(recs)}
    out = Parallel(n_jobs=8)(delayed(one_record)(
        r, tuple(arr[k][pos[r]] for k in ("fhr", "fhr_mask", "uc", "uc_mask")),
        int(ws[r]), int(ann_len[r]), ev[ev.rec == r]) for r in use)
    rows = [o[0] for o in out if o is not None]
    pairs = [p for o in out if o is not None for p in o[1]]
    df = pd.DataFrame(rows).set_index("rec")
    info["n_evaluated"] = len(df)
    info["covered_min_median"] = float(df.covered_min.median())
    df.to_csv(OUT / "fidelity_ctu_per_record.csv")

    res = {"info": info, "events": {}}
    for lab in ("ACC", "DEC", "UC"):
        for tag in ("any", "iou50"):
            pr = df[[f"{lab}_{tag}_tp", f"{lab}_{tag}_n_ref", f"{lab}_{tag}_n_det"]]
            pr.columns = ["tp", "n_ref", "n_det"]
            res["events"][f"{lab}_{tag}"] = boot_prf(pr, rng)
        rho = spearmanr(df[f"{lab}_any_n_ref"], df[f"{lab}_any_n_det"])
        res["events"][f"{lab}_count_spearman"] = round(float(rho.statistic), 4)
        res["events"][f"{lab}_count_per_window_median_expert_vs_detector"] = [
            float(df[f"{lab}_any_n_ref"].median()), float(df[f"{lab}_any_n_det"].median())]

    cm = pd.crosstab(pd.Series([p[0] for p in pairs], name="expert"),
                     pd.Series([p[1] for p in pairs], name="detector"))
    cm.to_csv(OUT / "fidelity_dec_type_confusion.csv")
    res["dec_type_confusion"] = cm.to_dict()
    res["dec_type_agreement_by_expert_type"] = {
        t: round(float(cm.loc[t, t] / cm.loc[t].sum()), 4) if t in cm.columns and t in cm.index else 0.0
        for t in cm.index}

    # weak labels (recomputed in the benchmark pipeline, read from csv) vs expert-derived concepts
    weak = pd.read_csv(ROOT / "results" / "features" / "figo_W60.csv", dtype={"record": str}).set_index("record")
    res["concepts"] = {}
    for c in CONCEPTS:
        y = df[f"exp_{c}"].astype(int)
        w = weak.loc[df.index, f"c_{c}"].astype(int)
        tp, tn = int(((y == 1) & (w == 1)).sum()), int(((y == 0) & (w == 0)).sum())
        kap = cohen_kappa_score(y, w) if y.nunique() > 1 or w.nunique() > 1 else np.nan
        bk = []
        idx = np.arange(len(y))
        for _ in range(N_BOOT):
            b = rng.integers(0, len(y), len(y))
            yb, wb = y.values[b], w.values[b]
            bk.append(cohen_kappa_score(yb, wb) if len(set(yb) | set(wb)) > 1 else np.nan)
        res["concepts"][c] = {
            "prev_expert": round(float(y.mean()), 4), "prev_weak": round(float(w.mean()), 4),
            "se_weak": round(tp / max(1, int(y.sum())), 4), "sp_weak": round(tn / max(1, int((y == 0).sum())), 4),
            "kappa": [round(float(kap), 4), round(float(np.nanpercentile(bk, 2.5)), 4),
                      round(float(np.nanpercentile(bk, 97.5)), 4)]}
        _ = idx

    # weak-label NeSy concept heads
    res["nesy_heads"] = {}
    for exp in ("nesy_ens_W60", "nesy_nocsup_W60"):
        files = sorted((ROOT / "results" / "cv" / exp).glob("r*_f*.npz"))
        if not files:
            continue
        acc = {}
        for f in files:
            d = np.load(f)
            if "x_c_bits" not in d.files:
                continue
            for r, v in zip(d["records_te"], d["x_c_bits"]):
                acc.setdefault(str(r), []).append(v)
        P = pd.DataFrame({r: np.mean(v, 0) for r, v in acc.items()}, index=BINARY).T
        P = P.loc[P.index.intersection(df.index)]
        rr = {}
        for c in CONCEPTS:
            ye, yw = df.loc[P.index, f"exp_{c}"], weak.loc[P.index, f"c_{c}"].astype(int)
            rr[c] = {"auroc_vs_expert": round(float(roc_auc_score(ye, P[c])), 4) if ye.nunique() > 1 else None,
                     "auroc_vs_weak": round(float(roc_auc_score(yw, P[c])), 4) if yw.nunique() > 1 else None,
                     "n": int(len(P))}
        res["nesy_heads"][exp] = rr
    return res


# ----------------------------------------------------------------------------- FHRMA
def _intervals(a, n):
    a = np.asarray(a, float)
    if a.size == 0:
        return []
    a = a.reshape(-1, 2)
    return [(int(round(s * 60 * FS)), min(n, int(round(e * 60 * FS)))) for s, e in a]


def fhrma(rng) -> dict:
    E = {k: v for k, v in A.fhrma_expert().items() if k.startswith("train")}
    rows, bl_err = [], []
    for name in sorted(E):
        v = E[name]
        sig = A.fhrma_signal(name)
        n = len(sig["fhr1"])
        cf = clean_fhr(sig["fhr1"])
        fhr, valid = np.nan_to_num(cf["fhr"], nan=0.0), cf["mask"].copy()
        excl = np.zeros(n, bool)
        for s, e in _intervals(v["notToAnalyse"], n) + _intervals(v["unreliableSignal"], n):
            excl[s:e] = True
        base = figo.baseline_curve(fhr, valid)
        accs = figo.detect_events(fhr, valid, base, +1)
        decs = figo.detect_events(fhr, valid, base, -1)
        keep = lambda s, e: not excl[s:e].all()  # noqa: E731
        row = {"rec": name}
        for lab, det, field in (("ACC", accs, "accelerations"), ("DEC", decs, "decelerations")):
            ex = [(s, e) for s, e in _intervals(v[field], n) if e > s and keep(s, e)]
            dt = [(d["start"], d["end"]) for d in det if keep(d["start"], d["end"])]
            for tag, thr in (("any", 0.0), ("iou50", 0.5)):
                row.update({f"{lab}_{tag}_{k}": x for k, x in counts(ex, dt, thr).items()})
        eb = np.asarray(v["baseline"], float)
        m = valid & ~excl & np.isfinite(base) & np.isfinite(eb) & (eb > 0)
        if m.any():
            d = base[m] - eb[m]
            row.update(bl_mae=float(np.abs(d).mean()), bl_rmsd=float(np.sqrt((d ** 2).mean())),
                       bl_bias=float(d.mean()), bl_n=int(m.sum()))
            bl_err.append(d)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("rec")
    df.to_csv(OUT / "fidelity_fhrma_per_record.csv")
    res = {"n_records": len(df), "events": {}}
    for lab in ("ACC", "DEC"):
        for tag in ("any", "iou50"):
            pr = df[[f"{lab}_{tag}_tp", f"{lab}_{tag}_n_ref", f"{lab}_{tag}_n_det"]]
            pr.columns = ["tp", "n_ref", "n_det"]
            res["events"][f"{lab}_{tag}"] = boot_prf(pr, rng)
    allerr = np.concatenate(bl_err)
    res["baseline"] = {"pooled_mae_bpm": round(float(np.abs(allerr).mean()), 3),
                       "pooled_rmsd_bpm": round(float(np.sqrt((allerr ** 2).mean())), 3),
                       "pooled_bias_bpm": round(float(allerr.mean()), 3),
                       "per_record_mae_median": round(float(df.bl_mae.median()), 3),
                       "per_record_mae_p90": round(float(df.bl_mae.quantile(0.9)), 3),
                       "frac_samples_abs_err_gt_5bpm": round(float((np.abs(allerr) > 5).mean()), 4)}
    return res


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"protocol": "see module docstring", "seed": SEED, "n_boot": N_BOOT,
           "ctu_uhb_vs_romagnoli": ctu_uhb(rng), "fhrma": fhrma(rng)}
    (OUT / "concept_fidelity.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
