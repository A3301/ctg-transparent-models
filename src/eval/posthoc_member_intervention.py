"""Post hoc analysis: apply the expert-concept intervention to each saved NeSy-E2E member (deviation #19).

Run: python -m src.eval.posthoc_member_intervention   (with the project .venv)
Output: results/study2/posthoc_member_intervention.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.data.concept_targets import load_targets
from src.data.inputs import signal_inputs
from src.models.nesy_v2 import EXPERT_BITS, EXPERT_IDX, NeSyV2

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"all": list(range(len(EXPERT_IDX))), "acc": [EXPERT_BITS.index("accel")],
        "dec": [i for i, name in enumerate(EXPERT_BITS) if name != "accel"]}


def run():
    torch.set_num_threads(2)
    records, signals = signal_inputs("W60", 1)
    targets = load_targets()
    assert list(records) == list(targets["records"])
    pos = {str(rec): i for i, rec in enumerate(records)}
    rows, errors = [], []
    n_members = 0
    directory = ROOT / "results/cv/nesy2_W60"
    for fold in sorted(directory.glob("r*_f*.npz")):
        with np.load(fold, allow_pickle=True) as z:
            indices = np.array([pos[str(r)] for r in z["records_te"]])
            eligible = np.flatnonzero(targets["annotated"][indices] & targets["rec_quality"][indices])
            keep = indices[eligible]
            data = torch.as_tensor(signals[keep])
            expert = torch.as_tensor(targets["win_masked"][keep])
            assert torch.all((expert == 0) | (expert == 1))
            predictions = {key: [] for key in ARMS}
            for m in range(5):
                ck = torch.load(directory / "weights" / f"{fold.stem}_m{m}.pt", map_location="cpu", weights_only=True)
                net = NeSyV2(4, **{k: v for k, v in ck["cfg"].items() if k in ("width", "n_stages", "kernel", "drop")})
                net.load_state_dict(ck["state"])
                net.eval()
                shift = np.log(ck["pos_weight"])
                with torch.inference_mode():
                    out = net(data)
                    p0 = expit((out["logit"].numpy() - shift) / ck["T"])
                    for key, cols in ARMS.items():
                        bits = out["lbits"].clone()
                        bits[:, [EXPERT_IDX[j] for j in cols]] = torch.where(expert[:, cols] > 0, 9.2, -9.2)
                        logit, _ = net.logic(out["lb"], out["lv"], bits)
                        predictions[key].append(expit((logit.numpy() - shift) / ck["T"]))
                errors.extend(np.abs(p0 - z["mem_te"][m][eligible]))
                n_members += 1
            means = [np.mean(predictions[key], axis=0) for key in ARMS]
            rows.extend(zip(z["records_te"][eligible], z["y_te"][eligible], z["p_te"][eligible], *means))
    d = pd.DataFrame(rows, columns=["record", "y", "original", *ARMS]).groupby("record").mean()
    y = d.y.to_numpy()
    original = d.original.to_numpy()
    assert n_members == 250 and len(y) == 340 and y.sum() == 16
    rng = np.random.default_rng(7511)
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    samples = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(2000)]
    auc_boot = np.array([roc_auc_score(y[ix], original[ix]) for ix in samples])
    auc0 = roc_auc_score(y, original)
    result = dict(members=n_members, n=len(y), events=int(y.sum()), baseline_auc=auc0,
                  baseline_member_mean_abs_error=float(np.mean(errors)), baseline_member_max_abs_error=float(np.max(errors)),
                  changes={})
    for key in ARMS:
        p = d[key].to_numpy()
        delta = np.abs(p - original)
        auc = roc_auc_score(y, p)
        interval = np.quantile([roc_auc_score(y[ix], p[ix]) - a for ix, a in zip(samples, auc_boot)], [.025, .975])
        result["changes"][key] = dict(auc=auc, delta_auc=auc-auc0, ci95_auc=interval.tolist(),
                                      mean_abs_risk=float(np.mean(delta)), percentiles_abs_risk=np.percentile(delta, [50, 90, 95, 99]).tolist(),
                                      n_abs_risk_ge_5pp=int((delta >= .05).sum()))
    use = np.array([i for i in range(len(records)) if str(records[i]) in {str(r[0]) for r in rows}])
    result["coverage"] = {"eligible_for_intervention": len(use), "positive_q_recordings": int(targets["q"][use].any(1).sum()),
                          "q_hours": float(targets["q"][use].sum() * 4/3600),
                          "q_hours_denominator": len(use), "q_records_with_acidaemia": int(((targets["q"][use].any(1)) & (np.isin(records[use], d.index[y == 1]))).sum())}
    return result


if __name__ == "__main__":
    result = run()
    (ROOT / "results/study2/posthoc_member_intervention.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
