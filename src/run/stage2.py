"""Plan B stage 2 (docs/analiz_plani_v3.md §4.2): detector events -> FIGO features -> transparent GAM.

For every outer fold of results/cv/det_W60:
  * outer-train records: events from the 3-fold cross-fitted single-member detectors (cf_prob, cf_thr),
  * outer-test records: events from the fold's 5-member detector ensemble with the fold's own
    validation-selected thresholds (x_f_prob, event_thr),
  * FIGO features recomputed with src.features.figo.extract(..., events=...) (rule baseline,
    FIGO timing rules for deceleration type), plus the Plan C HRV and quality variables,
  * `transpdet_gam` tuned in the inner CV of the outer-train records only (same code as Plan C).
Writes results/cv/transpdet_gam_W60/r{r}_f{f}.npz with the same keys the evaluation reads.

Run: .venv\\Scripts\\python -m src.run.stage2 [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.run.cv import SEED, fit_predict_sk, inner_select, transp_table  # noqa: E402

EXP, DET = "transpdet_gam", "det_W60"     # overridable: --exp transpdetk_gam --det detk_W60 (ek2)


def _events(p, thr):
    from src.eval.events import MIN_BINS, bins_to_events
    p = np.asarray(p, np.float32)
    return {"acc": bins_to_events(p[0], thr[0], MIN_BINS[0]), "dec": bins_to_events(p[1], thr[1], MIN_BINS[1]),
            "uc": bins_to_events(p[2], thr[2], MIN_BINS[2])}


def _features(i, fhr, fm, uc, um, p, thr):
    from src.data.inputs import FIGO_FEATURES
    from src.features.figo import extract
    f = extract(fhr, fm, uc, um, events=_events(p, thr))
    return i, np.array([f[k] for k in FIGO_FEATURES], np.float64)


def run(r, f, workers):
    from joblib import Parallel, delayed
    from src.data.inputs import FIGO_FEATURES, _load, labels
    out = ROOT / "results" / "cv" / f"{EXP}_W60" / f"r{r}_f{f}.npz"
    if out.exists():
        return str(out), "skip"
    t0 = time.time()
    z = np.load(ROOT / "results" / "cv" / DET / f"r{r}_f{f}.npz", allow_pickle=True)
    base = transp_table("transp_gam", "W60")             # FIGO (rule) + HRV + quality, 552 x 30
    recs = base.index.values.astype(str)
    pos = {k: i for i, k in enumerate(recs)}
    d = _load("W60")
    assert list(d["records"].astype(str)) == list(recs)
    jobs = [(pos[str(k)], z["cf_prob"][j], z["cf_thr"][int(z["cf_part"][j])]) for j, k in enumerate(z["cf_records"])]
    jobs += [(pos[str(k)], z["x_f_prob"][j], z["event_thr"]) for j, k in enumerate(z["records_te"])]
    res = Parallel(n_jobs=workers)(delayed(_features)(i, d["fhr"][i], d["fhr_mask"][i], d["uc"][i], d["uc_mask"][i], p, t)
                                   for i, p, t in jobs)
    X = base.values.astype(np.float64).copy()
    X[:, :len(FIGO_FEATURES)] = np.nan                    # only rows with detector features are ever used
    for i, v in res:
        X[i, :len(FIGO_FEATURES)] = v
    assert list(base.columns[:len(FIGO_FEATURES)]) == FIGO_FEATURES
    tr = np.array([pos[str(k)] for k in z["cf_records"]])
    te = np.array([pos[str(k)] for k in z["records_te"]])
    y = np.nan_to_num(labels("y_ph705").reindex(recs).values, nan=0).astype(np.int64)
    assert (y[te] == z["y_te"]).all()
    seed = SEED + 1000 * r + f
    cfg, cfgs, scores = inner_select(EXP, "sk", X, y, {}, tr, seed, None)
    fr = fit_predict_sk(EXP, cfg, X, y, tr, te, seed)
    val = fr["val_idx"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, records_te=recs[te], y_te=y[te], p_te=fr["mem_te"].mean(0), mem_te=fr["mem_te"],
             records_va=recs[val], y_va=y[val], p_va=fr["mem_va"].mean(0), cfg=json.dumps(cfg), cfgs=json.dumps(cfgs),
             inner_scores=json.dumps(scores), temps=json.dumps(fr["temps"]), seconds=time.time() - t0,
             det_features_te=X[te, :len(FIGO_FEATURES)].astype(np.float32))
    return str(out), f"ok {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exp", default=EXP)
    ap.add_argument("--det", default=DET)
    a = ap.parse_args()
    globals().update(EXP=a.exp, DET=a.det)
    (ROOT / "results" / "cv" / f"{EXP}_W60").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for r in range(5):
        for f in range(10):
            print(*run(r, f, a.workers), flush=True)
    print(f"done {EXP}_W60 in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
