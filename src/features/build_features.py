"""Extract FIGO features / weak concepts / FIGO class for every record and window.
Uses only the cleaned signals (no outcome information). Output: results/features/figo_<window>.csv"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.figo import extract  # noqa: E402

def run(window: str) -> pd.DataFrame:
    z = np.load(ROOT / "data" / "processed" / f"windows_{window}.npz")
    recs = z["records"]
    rows = Parallel(n_jobs=16)(delayed(extract)(z["fhr"][i], z["fhr_mask"][i], z["uc"][i], z["uc_mask"][i]) for i in range(len(recs)))
    df = pd.DataFrame(rows, index=pd.Index(recs, name="record"))
    out = ROOT / "results" / "features"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"figo_{window}.csv")
    return df

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["W60", "W30", "W90", "S1_60"]):
        df = run(w)
        print(w, df.shape)
        if w == "W60":
            print(df.describe().T[["mean", "50%", "min", "max"]].round(2).to_string())
            for c in [c for c in df.columns if c.startswith("c_") or c == "figo_class"]:
                print(c, df[c].value_counts(dropna=False).sort_index().to_dict())
