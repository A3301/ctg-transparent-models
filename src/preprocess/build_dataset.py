"""Clean all records and cut analysis windows.

Windows (Sig2Birth == 0 for every record, so signal end == birth):
  W60   last 60 min before birth (primary)
  W30   last 30 min
  W90   last 90 min (left-padded with missing when the record is shorter)
  S1_60 last 60 min of stage I (ending at Pos. II.st.; whole-signal end if no stage II)

Saved to data/processed/windows_<name>.npz with float32 arrays at 4 Hz:
  fhr, fhr_mask, uc, uc_mask  -> model inputs (after per-fold normalisation)
  stage2                      -> stage-II indicator. NOT A MODEL INPUT (leaks delivery mode /
                                 stage-II duration, see docs/faz_0_rapor.md). Stratification only.
  records                     -> record ids
The quality table (no outcome columns) goes to results/preprocess/window_quality.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.ctu_uhb import load_metadata, load_signals  # noqa: E402
from src.preprocess.clean import FS, _runs, clean_fhr, clean_uc  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "preprocess"
WINDOWS = {"W60": 60, "W30": 30, "W90": 90, "S1_60": 60}
EXCL_FHR_MISSING = 0.50  # exclusion threshold, applied on W60 (see analysis plan)


def cut(arr, start, end, fill):
    n = end - start
    out = np.full(n, fill, dtype=np.float32)
    s = max(start, 0)
    out[s - start:] = arr[s:end]
    return out


def longest_gap_s(mask):
    r = _runs(~mask.astype(bool))
    return max([(e - s) for s, e in r], default=0) / FS


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    store = {w: {k: [] for k in ("fhr", "fhr_mask", "uc", "uc_mask", "stage2")} for w in WINDOWS}
    qrows, crows = [], []
    for rec in meta.index:
        fhr, uc = load_signals(rec)
        cf, cu = clean_fhr(fhr), clean_uc(uc)
        n = len(fhr)
        crows.append({"record": rec, **{f"fhr_{k}": v for k, v in cf["stats"].items()},
                      **{f"uc_{k}": v for k, v in cu["stats"].items()}})
        pos2 = int(meta.at[rec, "pos_stage2"])
        s2 = np.zeros(n, np.float32)
        if 0 < pos2 < n:
            s2[pos2:] = 1
        for w, minutes in WINDOWS.items():
            L = int(minutes * 60 * FS)
            end = (pos2 if 0 < pos2 < n else n) if w == "S1_60" else n
            start = end - L
            f = cut(np.nan_to_num(cf["fhr"], nan=0.0), start, end, 0.0)
            fm = cut(cf["mask"].astype(np.float32), start, end, 0.0)
            u = cut(np.nan_to_num(cu["uc"], nan=0.0), start, end, 0.0)
            um = cut(cu["mask"].astype(np.float32), start, end, 0.0)
            st = cut(s2, start, end, 0.0)
            for k, v in zip(("fhr", "fhr_mask", "uc", "uc_mask", "stage2"), (f, fm, u, um, st)):
                store[w][k].append(v)
            qrows.append({"record": rec, "window": w, "start": start, "end": end,
                          "padded_s": max(0, -start) / FS,
                          "fhr_missing": float(1 - fm.mean()), "uc_missing": float(1 - um.mean()),
                          "fhr_longest_gap_s": longest_gap_s(fm), "uc_longest_gap_s": longest_gap_s(um),
                          "stage2_min_in_window": float(st.sum() / FS / 60)})
    for w in WINDOWS:
        np.savez_compressed(PROC / f"windows_{w}.npz", records=np.array(meta.index, dtype="U8"),
                            **{k: np.stack(v) for k, v in store[w].items()})
    q = pd.DataFrame(qrows)
    q.to_csv(OUT / "window_quality.csv", index=False)
    pd.DataFrame(crows).to_csv(OUT / "cleaning_stats.csv", index=False)

    w60 = q[q.window == "W60"].set_index("record")
    excluded = w60.index[w60.fhr_missing > EXCL_FHR_MISSING].tolist()
    (OUT / "exclusions.json").write_text(json.dumps({
        "rule": f"FHR missing fraction in W60 after cleaning > {EXCL_FHR_MISSING}",
        "n_excluded": len(excluded), "excluded": excluded,
        "n_included": int(len(w60) - len(excluded))}, indent=2))
    summ = q.groupby("window")[["fhr_missing", "uc_missing", "fhr_longest_gap_s", "stage2_min_in_window", "padded_s"]].describe(percentiles=[.5, .9]).round(3)
    summ.to_csv(OUT / "window_quality_summary.csv")
    cs = pd.DataFrame(crows).drop(columns="record").mean().round(5)
    print(cs.to_string())
    print(q.groupby("window")[["fhr_missing", "uc_missing", "stage2_min_in_window", "padded_s"]].median())
    for thr in (0.3, 0.4, 0.5):
        print(f"W60 fhr_missing>{thr}:", int((w60.fhr_missing > thr).sum()),
              " S1_60:", int((q[q.window == "S1_60"].fhr_missing > thr).sum()))
    print("excluded:", len(excluded))


if __name__ == "__main__":
    main()
