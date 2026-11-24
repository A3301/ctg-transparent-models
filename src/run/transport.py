"""Plan C transportability (docs/analiz_plani_v3_ek3.md): train on one group, test on another.

C1: train on vaginal deliveries (records 1001-1506), test on caesareans (2001-2046).
C3: train on FHR loss <= median (W60), test on FHR loss > median.
Same protocol as the CV runner (inner tuning, 15% calibration split, prior correction, temperature
scaling); 5 repeats with different seeds; test predictions averaged over repeats. Descriptive only.

Run: python -m src.run.transport [--models transp_lr,transp_gam,figo_lr,georgoulas,cnn]
Output: results/transport/<split>_<model>.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.run.cv import SEED, fit_predict_dl, fit_predict_sk, inner_select, model_inputs  # noqa: E402

OUT = ROOT / "results" / "transport"


def splits(recs):
    import json
    excl = set(json.loads((ROOT / "results/preprocess/exclusions.json").read_text())["excluded"])
    ok = np.array([r not in excl for r in recs])
    rid = np.array([int(r) for r in recs])
    wq = pd.read_csv(ROOT / "results/preprocess/window_quality.csv", dtype={"record": str})
    miss = wq[wq.window == "W60"].set_index("record").reindex(recs)["fhr_missing"].values
    med = np.median(miss[ok])
    return {"C1": (np.where(ok & (rid < 2000))[0], np.where(ok & (rid >= 2000))[0]),
            "C3": (np.where(ok & (miss <= med))[0], np.where(ok & (miss > med))[0])}


def run(model, split, n_rep=5):
    from src.data.inputs import labels
    kind_model = "cnn" if model == "cnn" else model
    recs, X, extra, hz, kind = model_inputs(kind_model, "W60")
    recs = np.asarray(recs).astype(str)
    y = np.nan_to_num(labels("y_ph705").reindex(recs).values, nan=0).astype(np.int64)
    tr, te = splits(recs)[split]
    out = OUT / f"{split}_{model}.npz"
    if out.exists():
        return str(out), "skip"
    t0 = time.time()
    P = []
    for r in range(n_rep):
        seed = SEED + 7000 + 100 * r + (0 if split == "C1" else 50)
        cfg, _, _ = inner_select(kind_model, kind, X, y, extra, tr, seed, "cuda")
        if kind == "dl":
            res = fit_predict_dl(kind_model, cfg, X, y, extra, tr, te, seed, "cuda", 5)
        else:
            res = fit_predict_sk(kind_model, cfg, X, y, tr, te, seed)
        P.append(res["mem_te"].mean(0))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, records_te=recs[te], y_te=y[te], p_te=np.mean(P, 0), p_rep=np.array(P),
             n_train=len(tr), events_train=int(y[tr].sum()))
    return str(out), f"ok {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="transp_lr,transp_gam,figo_lr,georgoulas,cnn")
    a = ap.parse_args()
    for split in ("C1", "C3"):
        for m in a.models.split(","):
            print(split, m, *run(m, split), flush=True)
    print("TRANSPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
