"""Case-level explanations and concept fidelity for the neuro-symbolic model."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.inputs import figo_table  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models.nesy import BASELINE, BINARY, CONCEPT_KEYS, RULES, VARIAB  # noqa: E402

PRETTY = {"accel": "accelerations present", "rep_any": "recurrent decelerations (>50% of contractions)",
          "rep_late": "recurrent late decelerations", "rep_variable": "recurrent variable decelerations",
          "early": "early decelerations", "prolonged": "prolonged deceleration (>3 min)",
          "very_prolonged": "deceleration >5 min", "reduced_var_long": "reduced variability >50 min",
          "increased_var_long": "increased (saltatory) variability >30 min",
          "rep_late_long": "repetitive late/prolonged decelerations >30 min"}
RULE_TXT = {"leak": "baseline risk", "pathological": "FIGO pathological pattern", "suspicious": "FIGO suspicious pattern",
            "rep_late": "recurrent late decelerations", "rep_variable": "recurrent variable decelerations",
            "prolonged": "prolonged deceleration", "tachycardia": "fetal tachycardia",
            "no_accel": "absence of accelerations", "reduced_variability": "reduced variability",
            "rep_any": "recurrent decelerations"}


def load_case_outputs(exp="nesy_ens_W60") -> tuple[pd.DataFrame, np.ndarray]:
    rows, rws = [], []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / exp / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        if "rule_w" in z.files and z["rule_w"].size:
            rws.append(z["rule_w"].mean(0))
        for i, rec in enumerate(z["records_te"]):
            rows.append(dict(record=str(rec), y=int(z["y_te"][i]), p=float(z["p_te"][i]),
                             unc_epi=float(z["unc_epi_te"][i]),
                             **{f"b_{n}": float(z["x_c_b"][i, j]) for j, n in enumerate(BASELINE)},
                             **{f"v_{n}": float(z["x_c_v"][i, j]) for j, n in enumerate(VARIAB)},
                             **{f"c_{n}": float(z["x_c_bits"][i, j]) for j, n in enumerate(BINARY)},
                             **{f"figo_{n}": float(z["x_figo_probs"][i, j]) for j, n in enumerate(("normal", "suspicious", "pathological"))}))
    df = pd.DataFrame(rows).groupby("record").mean(numeric_only=True)
    return df, (np.array(rws).mean(0) if rws else np.full(len(RULES), np.nan))


def rule_body_probs(r: pd.Series) -> dict:
    return {"leak": 1.0, "pathological": r.figo_pathological, "suspicious": r.figo_suspicious,
            "rep_late": r.c_rep_late, "rep_variable": r.c_rep_variable, "prolonged": r.c_prolonged,
            "tachycardia": r.b_tachy, "no_accel": 1 - r.c_accel, "reduced_variability": r.v_reduced,
            "rep_any": r.c_rep_any}


def explain(r: pd.Series, w: np.ndarray, unc_cut: float | None = None) -> str:
    body = rule_body_probs(r)
    contrib = {k: w[RULES.index(k)] * body[k] for k in RULES if k != "leak"}
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    reassure = []
    if r.b_normal > 0.7:
        reassure.append(f"normal baseline (P={r.b_normal:.2f})")
    if r.v_normal > 0.7:
        reassure.append(f"normal variability (P={r.v_normal:.2f})")
    if r.c_accel > 0.7:
        reassure.append(f"accelerations present (P={r.c_accel:.2f})")
    defer = "" if unc_cut is None else (" The model's epistemic uncertainty is HIGH for this trace; it is referred to the clinician." if r.unc_epi > unc_cut else " The model's epistemic uncertainty is low.")
    s = (f"Estimated risk of cord acidaemia (pH<7.05): {100 * r.p:.0f}%. "
         f"FIGO 2015 interpretation by the model: normal {r.figo_normal:.2f}, suspicious {r.figo_suspicious:.2f}, "
         f"pathological {r.figo_pathological:.2f}. Main drivers: "
         + "; ".join(f"{RULE_TXT[k]} (rule weight {w[RULES.index(k)]:.2f} × P={body[k]:.2f})" for k, _ in top) + ". "
         + (("Reassuring features: " + ", ".join(reassure) + ".") if reassure else "No strongly reassuring features.")
         + defer)
    return s


def concept_fidelity(df: pd.DataFrame, window="W60") -> pd.DataFrame:
    t = figo_table(window).reindex(df.index)
    rows = []
    for n in BINARY:
        lab = t[CONCEPT_KEYS[n]].values.astype(float)
        ok = np.isfinite(lab)
        k = int(np.nansum(lab[ok]))
        auc = M.auroc(lab[ok], df[f"c_{n}"].values[ok]) if 0 < k < ok.sum() else np.nan
        rows.append(dict(concept=n, n_weak_pos=k, auroc_vs_weak=auc, mean_prob=float(df[f"c_{n}"].mean())))
    for name, keys, col in (("baseline", BASELINE, "c_baseline"), ("variability", VARIAB, "c_variability")):
        lab = t[col].values
        ok = np.isfinite(lab)
        pred = df[[f"{name[0]}_{k}" for k in keys]].values.argmax(1)
        rows.append(dict(concept=name, n_weak_pos=int(ok.sum()), auroc_vs_weak=np.nan,
                         mean_prob=float((pred[ok] == lab[ok]).mean()), note="accuracy vs weak label"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df, w = load_case_outputs()
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rule": RULES, "weight_mean": w}).to_csv(out / "rule_weights.csv", index=False)
    concept_fidelity(df).to_csv(out / "concept_fidelity.csv", index=False)
    print(pd.DataFrame({"rule": RULES, "w": np.round(w, 3)}).to_string(index=False))
    print(concept_fidelity(df).round(3).to_string(index=False))
