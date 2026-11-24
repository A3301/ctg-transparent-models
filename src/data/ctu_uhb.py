"""Loader for the CTU-UHB intrapartum CTG database.

Separates header fields into explicit groups so that outcome / post-delivery
variables can never be silently used as model inputs (see LEAKAGE_FIELDS).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data" / "raw" / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
FS = 4.0  # Hz

# Header key -> column name
FIELD_MAP = {
    "pH": "ph", "BDecf": "bdecf", "pCO2": "pco2", "BE": "be",
    "Apgar1": "apgar1", "Apgar5": "apgar5",
    "NICU days": "nicu_days", "Seizures": "seizures", "HIE": "hie",
    "Intubation": "intubation", "Main diag.": "main_diag", "Other diag.": "other_diag",
    "Gest. weeks": "gest_weeks", "Weight(g)": "weight_g", "Sex": "sex",
    "Age": "mat_age", "Gravidity": "gravidity", "Parity": "parity",
    "Diabetes": "diabetes", "Hypertension": "hypertension", "Preeclampsia": "preeclampsia",
    "Liq. praecox": "liq_praecox", "Pyrexia": "pyrexia", "Meconium": "meconium",
    "Presentation": "presentation", "Induced": "induced",
    "I.stage": "stage1_min", "NoProgress": "no_progress", "CK/KP": "ck_kp",
    "II.stage": "stage2_min", "Deliv. type": "deliv_type",
    "dbID": "db_id", "Rec. type": "rec_type", "Pos. II.st.": "pos_stage2", "Sig2Birth": "sig2birth",
}

# Post-delivery / outcome-affected variables: NEVER model inputs.
LEAKAGE_FIELDS = [
    "ph", "bdecf", "pco2", "be", "apgar1", "apgar5",
    "nicu_days", "seizures", "hie", "intubation", "main_diag", "other_diag",
    "weight_g", "sex",  # known only at/after birth
    "deliv_type", "stage1_min", "stage2_min", "no_progress", "ck_kp", "meconium",
]
# Pre-delivery clinical priors: allowed ONLY in a separate, flagged experiment.
CLINICAL_PRIOR_FIELDS = [
    "gest_weeks", "mat_age", "gravidity", "parity", "diabetes", "hypertension",
    "preeclampsia", "liq_praecox", "pyrexia", "presentation", "induced",
]
# Signal descriptors (not outcomes; used for windowing / stratification).
SIGNAL_FIELDS = ["rec_type", "pos_stage2", "sig2birth", "n_samples", "duration_min"]


def list_records(db_dir: Path = DB_DIR) -> list[str]:
    return [r.strip() for r in (db_dir / "RECORDS").read_text().split() if r.strip()]


def parse_header(rec: str, db_dir: Path = DB_DIR) -> dict:
    lines = (db_dir / f"{rec}.hea").read_text().splitlines()
    first = lines[0].split()
    meta = {"record": rec, "n_samples": int(first[3])}
    meta["duration_min"] = meta["n_samples"] / FS / 60.0
    for ln in lines:
        if not ln.startswith("#") or ln.startswith("#-"):
            continue
        body = ln[1:]
        for key, col in FIELD_MAP.items():
            m = re.match(rf"^{re.escape(key)}\s+(\S+)\s*$", body)
            if m:
                val = m.group(1)
                try:
                    meta[col] = float(val)
                except ValueError:
                    meta[col] = np.nan
                break
    return meta


def load_metadata(db_dir: Path = DB_DIR) -> pd.DataFrame:
    df = pd.DataFrame([parse_header(r, db_dir) for r in list_records(db_dir)])
    return df.set_index("record")


def load_signals(rec: str, db_dir: Path = DB_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (fhr_bpm, uc) as float arrays at 4 Hz, using WFDB physical units."""
    import wfdb

    r = wfdb.rdrecord(str(db_dir / rec))
    names = [n.upper() for n in r.sig_name]
    fhr = r.p_signal[:, names.index("FHR")].astype(float)
    uc = r.p_signal[:, names.index("UC")].astype(float)
    return fhr, uc


def make_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Primary and secondary record-level labels. NaN outcome -> NaN label."""
    lab = pd.DataFrame(index=meta.index)

    def thr(col, f):
        s = meta[col]
        return f(s).astype(float).where(s.notna())

    lab["y_ph705"] = thr("ph", lambda s: s < 7.05)           # primary
    lab["y_ph715"] = thr("ph", lambda s: s <= 7.15)
    lab["y_ph705le"] = thr("ph", lambda s: s <= 7.05)      # Georgoulas 2017 consistency check only
    lab["y_bdecf12"] = thr("bdecf", lambda s: s >= 12)
    lab["y_apgar5_lt7"] = thr("apgar5", lambda s: s < 7)
    return lab
