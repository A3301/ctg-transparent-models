"""Fidelity of the approximate concept-intervention test (deviation log #17).

The pre-specified intervention test (src/eval/study2_eval.py) rebuilds each fold's risk from member-averaged concept
logits and member-averaged rule weights. Here the same reconstruction WITHOUT intervention is compared with the risk
the model actually produced (recorded repeat-averaged, calibrated probabilities), on the same 340 annotated,
quality-eligible records. Descriptive only.

Output: results/study2/intervention_check.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.eval.study2_eval import SEED_CV, VAL_FRAC  # noqa: E402


def check(exp, T, pos, splits, lab_all):
    import torch
    from src.models.nesy import FIGOLogic
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
        zz = np.load(fp, allow_pickle=True)
        rr, ff = map(int, Path(fp).stem[1:].split("_f"))
        tr = np.array(splits["repeats"][rr][ff]["train"])
        ytr = lab_all.reindex(tr).fillna(0).astype(int).values
        fit, _ = train_test_split(np.arange(len(tr)), test_size=VAL_FRAC, stratify=ytr, random_state=SEED_CV + 1000 * rr + ff)
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
        with torch.no_grad():
            l0, _ = logic(lb[sel], lv[sel], lbits[sel])
        p0 = 1 / (1 + np.exp(-((l0.numpy() - shift) / Tm)))
        for k, a in zip(sel, p0):
            rows.append(dict(record=str(zz["records_te"][k]), y=int(zz["y_te"][k]), p_rebuilt=float(a), p_recorded=float(zz["p_te"][k])))
    d = pd.DataFrame(rows).groupby("record").agg(y=("y", "first"), p_rebuilt=("p_rebuilt", "mean"), p_recorded=("p_recorded", "mean"))
    return dict(n=len(d), n_pos=int(d.y.sum()), pearson=float(pearsonr(d.p_rebuilt, d.p_recorded)[0]),
                spearman=float(spearmanr(d.p_rebuilt, d.p_recorded)[0]),
                mean_abs_diff=float(np.mean(np.abs(d.p_rebuilt - d.p_recorded))),
                auroc_rebuilt=float(roc_auc_score(d.y, d.p_rebuilt)), auroc_recorded=float(roc_auc_score(d.y, d.p_recorded)))


def main():
    T = load_targets()
    pos = {str(r): i for i, r in enumerate(T["records"])}
    splits = json.loads((ROOT / "results" / "splits" / "outer_splits.json").read_text())
    lab_all = pd.read_csv(ROOT / "results" / "eda" / "labels.csv", dtype={"record": str}).set_index("record")["y_ph705"]
    out = {e: check(e, T, pos, splits, lab_all) for e in ("nesy2_W60", "nesy_ens_W60")}
    (ROOT / "results/study2/intervention_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    from src.utils.repro import log_experiment
    log_experiment("S2-check", "intervention_check", {"script": "src/eval/intervention_check.py", "deviation": "#17"}, 0, out,
                   "fidelity of the approximate intervention rebuild")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
