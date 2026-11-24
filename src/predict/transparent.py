"""Apply the released transparent linear model to new intrapartum CTG recordings (this paper,
Methods §2.4, "Released model").

Input: one CSV per recording with columns `fhr` (bpm) and `uc` (device units) sampled at 4 Hz, ending at
delivery (or the time point of interest). Missing samples may be empty/NaN or 0.
Processing is identical to the study: cleaning (src/preprocess/clean.py), last 60 min window, 23 FIGO
features (src/features/figo.py), 3 HRV features (src/features/hrv.py) and 4 signal-quality variables.
The model specification is read from results/study3/transp_lr_final_model.json.

The linear model was refitted on all 536 CTU-UHB recordings with the same prior correction but without
temperature scaling; its discrimination is estimated by the cross-validated performance of the same procedure.
The released specification is not externally validated and is not intended for clinical use. Recalibrate
on local outcome data before interpreting absolute risks (see the paper's Supplementary Material, "Full
specification of the refitted linear model").

Usage: python -m src.predict.transparent rec1.csv rec2.csv ... [--out predictions.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.figo import extract  # noqa: E402
from src.features.hrv import hrv_features  # noqa: E402
from src.preprocess.clean import clean_fhr, clean_uc  # noqa: E402

FS, WIN = 4, 60 * 60 * 4
SPEC = ROOT / "results" / "study3" / "transp_lr_final_model.json"


def longest_gap_s(mask):
    m = np.r_[False, ~mask.astype(bool), False].astype(int)
    d = np.diff(m)
    runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    return float(runs.max() / FS) if len(runs) else 0.0


def features(fhr_raw, uc_raw):
    fhr_raw = np.nan_to_num(np.asarray(fhr_raw, float), nan=0.0)
    uc_raw = np.nan_to_num(np.asarray(uc_raw, float), nan=0.0)
    cf, cu = clean_fhr(fhr_raw), clean_uc(uc_raw)
    f = np.nan_to_num(cf["fhr"], nan=0.0)[-WIN:]
    fm = cf["mask"].astype(np.float32)[-WIN:]
    u = np.nan_to_num(cu["uc"], nan=0.0)[-WIN:]
    um = cu["mask"].astype(np.float32)[-WIN:]
    if len(f) < WIN:                          # pad at the start, as in the study (padding counts as missing)
        pad = WIN - len(f)
        f, fm, u, um = (np.concatenate([np.zeros(pad, np.float32), a]) for a in (f, fm, u, um))
    x = extract(f, fm, u, um)
    x.update(hrv_features(f, fm))
    x.update(fhr_missing=float(1 - fm.mean()), uc_missing=float(1 - um.mean()),
             fhr_longest_gap_s=longest_gap_s(fm > 0.5), uc_longest_gap_s=longest_gap_s(um > 0.5))
    return x


def predict(x, spec):
    lp = spec["intercept_balanced"] - float(spec["prior_correction"].split("=")[1].split()[0])
    for c in spec["coefficients"]:
        v = x.get(c["variable"], np.nan)
        v = c["median_imputation"] if v is None or not np.isfinite(v) else v
        lp += c["coef_per_sd"] * (v - c["mean"]) / c["sd"]
    return 1 / (1 + np.exp(-lp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    rows = []
    for p in a.csv:
        d = pd.read_csv(p)
        x = features(d["fhr"].values, d["uc"].values)
        rows.append({"file": Path(p).name, "risk_ph_lt_7_05": predict(x, spec), "fhr_missing": x["fhr_missing"]})
    out = pd.DataFrame(rows)
    if a.out:
        out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
