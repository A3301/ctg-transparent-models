"""Generate the frozen outer CV splits (5 repeats x stratified 10-fold, record level).
Stratification: primary label x presence of stage II (stage II is NOT a model input)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SEED = 20261122
N_REPEATS, N_FOLDS = 5, 10


def load_population(include_excluded: bool = False) -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "results/eda/metadata_all.csv", dtype={"record": str}).set_index("record")
    lab = pd.read_csv(ROOT / "results/eda/labels.csv", dtype={"record": str}).set_index("record")
    df = meta[["pos_stage2"]].join(lab)
    df["has_stage2"] = (df["pos_stage2"] > 0).astype(int)
    if not include_excluded:
        ex = json.loads((ROOT / "results/preprocess/exclusions.json").read_text())["excluded"]
        df = df.drop(index=ex)
    return df


def make_splits(df: pd.DataFrame, label: str = "y_ph705") -> dict:
    d = df[df[label].notna()]
    strat = (d[label].astype(int) * 2 + d["has_stage2"]).values
    recs = d.index.values
    out = {"label": label, "seed": SEED, "n_records": int(len(d)), "repeats": []}
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r)
        folds = [{"train": recs[tr].tolist(), "test": recs[te].tolist()} for tr, te in skf.split(recs, strat)]
        out["repeats"].append(folds)
    return out


def main() -> None:
    out_dir = ROOT / "results" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, label, inc in [("outer_splits", "y_ph705", False),
                             ("outer_splits_ph715", "y_ph715", False),
                             ("outer_splits_bdecf12", "y_bdecf12", False),
                             ("outer_splits_apgar5", "y_apgar5_lt7", False),
                             ("outer_splits_all552", "y_ph705", True)]:
        s = make_splits(load_population(inc), label)
        (out_dir / f"{name}.json").write_text(json.dumps(s))
        tests = [len(f["test"]) for f in s["repeats"][0]]
        print(name, s["n_records"], "test sizes", min(tests), max(tests))


if __name__ == "__main__":
    main()
