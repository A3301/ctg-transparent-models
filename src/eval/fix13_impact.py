"""Old vs new results for the deceleration-area fix (deviation log #13): repeat-averaged out-of-fold AUROC and Brier
of every rerun experiment, from the parked old fold files (results/cv/_prefix13/) and the new ones (results/cv/).

Output: results/study3/fix13_impact.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def avg(folder):
    rows = []
    for fp in sorted(glob.glob(str(folder / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        rows += [dict(record=str(r), y=int(y), p=float(p)) for r, y, p in zip(z["records_te"], z["y_te"], z["p_te"])]
    if not rows:
        return None
    d = pd.DataFrame(rows).groupby("record").agg(y=("y", "first"), p=("p", "mean"))
    return dict(n=len(d), n_pos=int(d.y.sum()), auroc=float(roc_auc_score(d.y, d.p)), brier=float(brier_score_loss(d.y, d.p)))


def main():
    out = {}
    for old in sorted((ROOT / "results/cv/_prefix13").iterdir()):
        if not old.is_dir():
            continue
        new = ROOT / "results/cv" / old.name
        a, b = avg(old), avg(new)
        if a and b:
            out[old.name] = dict(old=a, new=b, d_auroc=b["auroc"] - a["auroc"], d_brier=b["brier"] - a["brier"])
    tr = {}
    for old in sorted((ROOT / "results/transport/_prefix13").glob("*.npz")):
        new = ROOT / "results/transport" / old.name
        if new.exists():
            zo, zn = np.load(old), np.load(new)
            tr[old.stem] = dict(old_auroc=float(roc_auc_score(zo["y_te"], zo["p_te"])), new_auroc=float(roc_auc_score(zn["y_te"], zn["p_te"])))
    out["_transport"] = tr
    (ROOT / "results/study3/fix13_impact.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    from src.utils.repro import log_experiment
    for k, v in out.items():
        if k == "_transport":
            continue
        log_experiment("S3-fix13", k, {"script": "scripts/run_fix13.sh", "deviation": "#13 deceleration-area fix" if "hrv_lr" not in k and "lr_no" not in k else "#16 descriptive additions"},
                       20261122, {"auroc_old": v["old"]["auroc"], "auroc_new": v["new"]["auroc"], "brier_new": v["new"]["brier"], "n": v["new"]["n"], "n_pos": v["new"]["n_pos"]},
                       "rerun with corrected FIGO features (results/study3/fix13_impact.json)")
    for k in ("hrv_lr_W60", "transp_lr_nohrv_W60", "transp_lr_noq_W60"):     # new experiments (deviation #16)
        v = avg(ROOT / "results/cv" / k)
        if v:
            log_experiment("S3-review", k, {"script": "scripts/run_fix13.sh", "deviation": "#16 descriptive additions"}, 20261122, v,
                           "post hoc descriptive model added after the primary results")
            print(f"{k:22s} AUROC {v['auroc']:.4f} (new experiment)")
    for k, v in out.items():
        if k != "_transport":
            print(f"{k:22s} AUROC {v['old']['auroc']:.4f} -> {v['new']['auroc']:.4f} ({v['d_auroc']:+.4f})")
    for k, v in tr.items():
        print(f"transport {k:22s} {v['old_auroc']:.4f} -> {v['new_auroc']:.4f}")


if __name__ == "__main__":
    main()
