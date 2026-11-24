"""Model inputs from the processed windows (fixed scaling constants -> no fold leakage).

FORBIDDEN_INPUTS lists header fields that must never enter a model; check_no_forbidden()
is called by every runner on the feature/column list it is about to use.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

FORBIDDEN_INPUTS = {
    "ph", "bdecf", "pco2", "be", "apgar1", "apgar5", "nicu_days", "seizures", "hie", "intubation",
    "main_diag", "other_diag", "weight_g", "sex", "deliv_type", "stage1_min", "stage2_min",
    "no_progress", "ck_kp", "meconium", "pos_stage2", "sig2birth", "duration_min", "n_samples",
    "stage2", "has_stage2", "stage2_min_in_window", "y_ph705", "y_ph715", "y_bdecf12", "y_apgar5_lt7", "y_ph705le",
}
FHR_C, FHR_S = 140.0, 30.0
UC_C, UC_S = 20.0, 30.0


def check_no_forbidden(columns) -> None:
    bad = sorted(set(map(str, columns)) & FORBIDDEN_INPUTS)
    assert not bad, f"FORBIDDEN model inputs: {bad}"


@lru_cache(maxsize=8)
def _load(window: str):
    z = np.load(PROC / f"windows_{window}.npz")
    return {k: z[k] for k in ("records", "fhr", "fhr_mask", "uc", "uc_mask")}


def _down(x, m, factor):
    n = x.shape[1] // factor
    x = x[:, :n * factor].reshape(len(x), n, factor)
    m = m[:, :n * factor].reshape(len(m), n, factor)
    s = m.sum(-1)
    xm = np.where(s > 0, (x * m).sum(-1) / np.maximum(s, 1), 0.0)
    return xm.astype(np.float32), (s / factor >= 0.5).astype(np.float32)


@lru_cache(maxsize=16)
def signal_inputs(window: str, hz: float = 1.0):
    """Return (records, X) with X (N, 4, T): [fhr, fhr_mask, uc, uc_mask] at `hz`."""
    d = _load(window)
    factor = int(round(4 / hz))
    f, fm = _down(d["fhr"], d["fhr_mask"], factor)
    u, um = _down(d["uc"], d["uc_mask"], factor)
    f = np.where(fm > 0, (f - FHR_C) / FHR_S, 0.0)
    u = np.where(um > 0, (u - UC_C) / UC_S, 0.0)
    X = np.stack([f, fm, u, um], 1).astype(np.float32)
    return d["records"].astype(str), X


@lru_cache(maxsize=8)
def quality_vector(window: str):
    """MCNN quality vector: valid-sample ratio of raw 4 Hz FHR in 15-min windows, 5-min step."""
    d = _load(window)
    m = d["fhr_mask"]
    L, step = 15 * 60 * 4, 5 * 60 * 4
    starts = range(0, m.shape[1] - L + 1, step)
    return np.stack([m[:, s:s + L].mean(1) for s in starts], 1).astype(np.float32)


def labels(label: str = "y_ph705") -> pd.Series:
    lab = pd.read_csv(ROOT / "results/eda/labels.csv", dtype={"record": str}).set_index("record")
    return lab[label]


def figo_table(window: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"results/features/figo_{window}.csv", dtype={"record": str}).set_index("record")


FIGO_FEATURES = ["baseline", "stv_ms", "ltv_ms", "var_amp_median", "reduced_var_min", "increased_var_min",
                 "n_acc", "n_dec", "n_dec_early", "n_dec_late", "n_dec_variable", "n_dec_prolonged",
                 "n_dec_other", "dec_total_min", "dec_max_s", "dec_area", "dec_max_depth",
                 "late_frac_contr", "variable_frac_contr", "early_frac_contr", "any_dec_frac_contr",
                 "sinusoidal_min", "uc_per_10min"]
CONCEPTS = {  # name -> number of classes (2 = binary)
    "c_baseline": 4, "c_variability": 4, "c_accel": 2, "c_rep_any": 2, "c_rep_late": 2,
    "c_rep_variable": 2, "c_early": 2, "c_prolonged": 2, "c_very_prolonged": 2,
    "c_reduced_var_long": 2, "c_increased_var_long": 2, "c_rep_late_long": 2,
}
