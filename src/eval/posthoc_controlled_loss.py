"""Post hoc analysis: paired controlled ablation of the acidaemia outcome loss (deviation #19).

Run: python -m src.eval.posthoc_controlled_loss [--device cuda]   (with the project .venv; GPU)
Output: results/study2/posthoc_controlled_loss.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from src.data.concept_targets import load_targets
from src.data.inputs import labels, signal_inputs
from src.eval.concept_fidelity import match
from src.eval.events import MIN_BINS, bins_to_events, expert_events, in_region, select_thresholds
from src.models.nesy_v2 import NeSyV2
from src.run.cv import SEED, VAL_FRAC, nesy2_targets
from src.train.trainer import predict_logits

ROOT = Path(__file__).resolve().parents[2]
CFG = dict(width=16, n_stages=3, kernel=7, drop=.3, lr=.001, wd=.001, bs=32)
EPOCHS = 24


class ControlledNeSy(NeSyV2):
    def __init__(self, outcome_weight):
        super().__init__(**CFG)
        self.outcome_weight = outcome_weight

    def loss(self, out, y, aux, pos_weight):
        total = super().loss(out, y, aux, pos_weight)
        if self.outcome_weight:
            return total
        outcome = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
        return total - outcome


def f1(counts):
    sums = counts.sum(axis=-2)
    return 2 * sums[..., 0] / np.maximum(sums[..., 1] + sums[..., 2], 1)


def run(device):
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    records, X = signal_inputs("W60", 1)
    pos = {str(r): i for i, r in enumerate(records)}
    y = labels("y_ph705").reindex(records).to_numpy(dtype=np.int64)
    target = load_targets()
    aux = nesy2_targets("nesy2", "W60", records)
    splits = json.loads((ROOT / "results/splits/outer_splits.json").read_text())["repeats"][0]
    rows = {}
    for fold, part in enumerate(splits):
        tr = np.array([pos[r] for r in part["train"]])
        te = np.array([pos[r] for r in part["test"]])
        seed = SEED + fold
        fit, val = train_test_split(tr, test_size=VAL_FRAC, stratify=y[tr], random_state=seed)
        assert not set(te).intersection(fit) and not set(te).intersection(val)
        paired = {}
        for weight in (0, 1):
            torch.manual_seed(seed)
            net = ControlledNeSy(weight)
            net = fit_fixed(net, X, y, aux, fit, seed, device)
            val_pred = predict_logits(net, X[val], device=device)["f_prob"]
            threshold, _ = select_thresholds(val_pred, val, target)
            test_pred = predict_logits(net, X[te], device=device)["f_prob"]
            paired[weight] = (threshold, test_pred)
        for j, i in enumerate(te):
            region = target["q"][i]
            vals = np.zeros((2, 2, 2, 3), int)
            for lab in (0, 1):
                ref = in_region(expert_events(target, i, lab), region)
                for weight, (threshold, prob) in paired.items():
                    detected = in_region(bins_to_events(prob[j, lab], threshold[lab], MIN_BINS[lab]), region)
                    for k, iou in enumerate((0., .5)):
                        vals[k, lab, weight] = [len(match(ref, detected, iou)), len(ref), len(detected)]
            assert i not in rows
            rows[i] = vals
        print(f"paired training and held-out evaluation completed for fold {fold + 1}/10", flush=True)
    counts = np.moveaxis(np.array([rows[i] for i in sorted(rows)]), 0, -2)
    assert counts.shape == (2, 2, 2, 536, 3)
    rng = np.random.default_rng(1265)
    indices = rng.integers(0, counts.shape[-2], size=(2000, counts.shape[-2]))
    result = {}
    for q, name in enumerate(("any_overlap", "iou_50")):
        result[name] = {}
        for lab, event in enumerate(("ACC", "DEC")):
            a, b = counts[q, lab]
            delta = f1(a[indices]) - f1(b[indices])
            result[name][event] = dict(no_outcome_f1=float(f1(a)), with_outcome_f1=float(f1(b)),
                                       difference=float(f1(a) - f1(b)), ci95=np.quantile(delta, [.025, .975]).tolist())
    return dict(configuration=CFG, epochs=EPOCHS, repeats=1, n_folds=10, n_records=len(rows), outcomes=result)


def fit_fixed(model, X, y, targets, fit, seed, device):
    model.to(device)
    model.train()
    values = torch.as_tensor(X[fit], device=device)
    outcome = torch.as_tensor(y[fit], dtype=torch.float32, device=device)
    aux = {k: torch.as_tensor(v[fit], device=device) for k, v in targets.items()}
    pos_weight = (len(fit) - outcome.sum()) / outcome.sum()
    opt = torch.optim.AdamW(model.param_groups(CFG))
    rng = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(fit), generator=rng).to(device).split(CFG["bs"]):
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = model.loss(model(values[idx]), outcome[idx], {k: v[idx] for k, v in aux.items()}, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "results/study2/posthoc_controlled_loss.json")
    args = parser.parse_args()
    result = run(args.device)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
