"""Post hoc analysis: validation-selected operating points, FHR-loss comparator and paired net-benefit
differences of the transparent models (deviation #20).

Run: python -m src.eval.posthoc_transparent   (with the project .venv)
Output: results/study3/posthoc_transparent.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.metrics import net_benefit

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"cnn": "cnn_ens_W60", "gam": "transp_gam_W60",
          "lr": "transp_lr_W60", "hrv_lr": "hrv_lr_W60", "figo_lr": "figo_lr_W60"}


def load_model(experiment):
    rows = []
    for path in sorted((ROOT / "results/cv" / experiment).glob("r*_f*.npz")):
        with np.load(path, allow_pickle=True) as z:
            assert not set(z["records_te"]).intersection(z["records_va"])
            threshold = np.quantile(z["p_va"][z["y_va"] == 0], .9, method="higher")
            rows.extend(zip(z["records_te"], z["y_te"], z["p_te"], z["p_te"] > threshold))
    d = pd.DataFrame(rows, columns=["record", "y", "p", "positive"])
    assert (d.groupby("record").size() == 5).all() and (d.groupby("record").y.nunique() == 1).all()
    return d.groupby("record").agg(y=("y", "first"), p=("p", "mean"), positive=("positive", "mean"))


def run():
    pred = {k: load_model(v) for k, v in MODELS.items()}
    y = pred["cnn"].y.to_numpy()
    assert len(y) == 536 and y.sum() == 38
    for d in pred.values():
        assert d.index.equals(pred["cnn"].index) and np.array_equal(d.y, y)
    points = {k: {"sensitivity": float(d.positive[y == 1].mean()), "specificity": float(1-d.positive[y == 0].mean())}
              for k, d in pred.items()}
    quality = pd.read_csv(ROOT / "results/preprocess/window_quality.csv", dtype={"record": str})
    quality = quality[quality.window.eq("W60")].set_index("record").loc[pred["cnn"].index]
    rng = np.random.default_rng(7560)
    indices = rng.integers(0, len(y), size=(2000, len(y)))
    risk = {k: d.p.to_numpy() for k, d in pred.items()}
    nb = {}
    for t in (.10, .20):
        nb[str(t)] = {}
        for comparator in ("cnn", "hrv_lr"):
            a, b = risk["lr"], risk[comparator]
            diff = net_benefit(y, a, [t])[0] - net_benefit(y, b, [t])[0]
            replicates = [net_benefit(y[ix], a[ix], [t])[0] - net_benefit(y[ix], b[ix], [t])[0] for ix in indices]
            nb[str(t)][comparator] = {"diff": float(diff), "ci95": np.quantile(replicates, [.025, .975]).tolist()}
    return {"n": len(y), "events": int(y.sum()), "validation_threshold_test_points": points,
            "missing_only_auc": roc_auc_score(y, quality.fhr_missing), "linear_minus_comparator_net_benefit": nb}


if __name__ == "__main__":
    result = run()
    (ROOT / "results/study3/posthoc_transparent.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
