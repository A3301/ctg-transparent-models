"""Expert concept targets and quality mask for the end-to-end model (docs/analiz_plani_v2.md §3).

Built once for all W60 records and cached to data/processed/expert_targets_W60.npz (git-ignored:
derived from Romagnoli 2020, CC BY-NC-ND). Targets are TRAINING TARGETS / REFERENCE ONLY, never
model inputs; src/run/cv.py blanks the rows of outer-test records before training (§7).

Per record (bins of 4 s; W60 -> 900 bins):
  frame_acc, frame_dec, frame_uc   uint8   1 if >= 50 % of the bin lies in an expert event
  frame_dtype                      int8    0..3 = E/L/V/P on DEC bins, -1 elsewhere
  q                                uint8   quality mask (§3.2): annotated record, covered bin,
                                           >= 90 % valid FHR in the centred 60 s, and record-level
                                           W60 FHR missing <= 20 %
  cov                              uint8   covered bin (annotation coverage) - used by the
                                           no-mask ablation, which treats every covered bin as known
  win_masked, win_nomask           int8    8 window concepts (EXPERT_BITS), 1/0/-1(unknown)
Expert event intervals (window samples at 4 Hz, clipped to coverage) are stored flat for event-level
evaluation: ev_rec (record index), ev_label (0 ACC, 1 DEC, 2 UC), ev_type (-1 or 0..3), ev_start, ev_end.

Run: .venv\\Scripts\\python -m src.data.concept_targets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.concept_fidelity import TYPE_MAP, expert_concepts  # noqa: E402
from src.features import figo  # noqa: E402

FS = 4
BIN = 4 * FS                      # 4 s at 4 Hz
OUT = ROOT / "data" / "processed" / "expert_targets_W60.npz"
LABELS = {"ACC": 0, "DEC": 1, "UC": 2}
DTYPES = ["E", "L", "V", "P"]
# order matches src.models.nesy.BINARY positions used for expert supervision
EXPERT_BITS = ["accel", "rep_any", "rep_late", "rep_variable", "early", "prolonged", "very_prolonged",
               "rep_late_long"]
Q_LOCAL_S, Q_LOCAL_FRAC, Q_RECORD_MISSING = 60, 0.90, 0.20


def rep_late_long_expert(decs, weak_row, win_min):
    """FIGO 'repetitive late/prolonged > 30 min (20 if reduced variability)'; variability from weak label."""
    lp = sorted([(s, e) for s, e, t in decs if t in ("late", "prolonged")])
    rep_late_or_prol = len(lp) > 0
    if len(lp) < 2 or not rep_late_or_prol:
        return 0
    span = (lp[-1][1] - lp[0][0]) / FS / 60
    reduced = (weak_row.get("c_reduced_var_long", 0) == 1) or (weak_row.get("c_variability", -1) == 0)
    lim = (20.0 if reduced else 30.0) * win_min / 60.0
    return int(span > lim)


def build() -> dict:
    z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
    recs = [str(r) for r in z["records"]]
    W = z["fhr"].shape[1]
    nb = W // BIN
    wq = pd.read_csv(ROOT / "results" / "preprocess" / "window_quality.csv", dtype={"record": str})
    wq = wq[wq.window == "W60"].set_index("record")
    ev = pd.read_csv(ROOT / "data" / "processed" / "romagnoli_events.csv", dtype={"subtype": str})
    ev["rec"] = ev["rec"].astype(str)
    ann_len = {str(k): int(v) for k, v in json.loads(
        (ROOT / "results" / "annotations" / "romagnoli_summary.json").read_text())["ann_len_by_record"].items()}
    weak = pd.read_csv(ROOT / "results" / "features" / "figo_W60.csv", dtype={"record": str}).set_index("record")
    annotated = set(ev.rec)

    n = len(recs)
    out = {k: np.zeros((n, nb), np.uint8) for k in ("frame_acc", "frame_dec", "frame_uc", "q", "cov")}
    out["frame_dtype"] = np.full((n, nb), -1, np.int8)
    out["win_masked"] = np.full((n, len(EXPERT_BITS)), -1, np.int8)
    out["win_nomask"] = np.full((n, len(EXPERT_BITS)), -1, np.int8)
    flat = {k: [] for k in ("ev_rec", "ev_label", "ev_type", "ev_start", "ev_end")}
    rec_quality = np.zeros(n, np.uint8)
    for i, r in enumerate(recs):
        ws = int(wq.at[r, "start"])
        c0, c1 = max(ws, 0) - ws, min(ws + W, ann_len.get(r, 0)) - ws
        fm = z["fhr_mask"][i] > 0.5
        # coverage per bin (>= 50 % of the bin covered)
        idx = np.arange(nb) * BIN
        covered = ((np.minimum(idx + BIN, c1) - np.maximum(idx, c0)) >= BIN / 2) if c1 > c0 else np.zeros(nb, bool)
        out["cov"][i] = covered
        # local quality: >= 90 % valid FHR in the centred 60 s (4 Hz mask)
        k = Q_LOCAL_S * FS
        cs = np.concatenate([[0], np.cumsum(fm)])
        centre = idx + BIN // 2
        lo, hi = np.clip(centre - k // 2, 0, W), np.clip(centre + k // 2, 0, W)
        local_ok = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1) >= Q_LOCAL_FRAC
        rq = float(wq.at[r, "fhr_missing"]) <= Q_RECORD_MISSING
        rec_quality[i] = rq
        is_ann = r in annotated
        out["q"][i] = covered & local_ok & is_ann & rq

        # expert events clipped to coverage, window coordinates
        decs, cons, accs = [], [], []
        for _, e in ev[ev.rec == r].iterrows():
            s, t = max(int(e.start) - ws, c0), min(int(e.end) - ws, c1)
            if t <= s or e.label not in LABELS:
                continue
            lab = LABELS[e.label]
            ty = DTYPES.index(e.subtype) if (e.label == "DEC" and e.subtype in DTYPES) else -1
            for key, v in (("ev_rec", i), ("ev_label", lab), ("ev_type", ty), ("ev_start", s), ("ev_end", t)):
                flat[key].append(v)
            # frame labels: bins with >= 50 % overlap
            b0, b1 = s // BIN, min(nb, (t + BIN - 1) // BIN)
            for b in range(b0, b1):
                ov = min(t, (b + 1) * BIN) - max(s, b * BIN)
                if ov >= BIN / 2:
                    out[["frame_acc", "frame_dec", "frame_uc"][lab]][i, b] = 1
                    if lab == 1 and ty >= 0:
                        out["frame_dtype"][i, b] = ty
            if lab == 0:
                accs.append((s, t))
            elif lab == 1:
                decs.append((s, t, TYPE_MAP.get(e.subtype, "unknown")))
            else:
                cons.append((s, t))

        exc = expert_concepts(decs, cons, z["fhr"][i], fm, z["uc"][i], z["uc_mask"][i] > 0.5)
        exc["accel"] = int(len(accs) > 0)
        exc["rep_late_long"] = rep_late_long_expert(decs, weak.loc[r].to_dict() if r in weak.index else {}, W / FS / 60)
        vals = np.array([exc[b] for b in EXPERT_BITS], np.int8)
        out["win_nomask"][i] = vals                                   # ablation: absent = negative
        if is_ann and rq:
            out["win_masked"][i] = vals
        else:                                                         # positives only; negatives unknown
            out["win_masked"][i] = np.where(vals == 1, 1, -1)
    res = {**out, **{k: np.array(v, np.int32) for k, v in flat.items()},
           "records": np.array(recs), "rec_quality": rec_quality,
           "annotated": np.array([r in annotated for r in recs], np.uint8),
           "expert_bits": np.array(EXPERT_BITS), "bin_samples": BIN}
    return res


def load_targets() -> dict:
    d = np.load(OUT, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    res = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **res)
    n = len(res["records"])
    print(f"{n} records -> {OUT}")
    print("annotated", int(res["annotated"].sum()), "record-quality ok", int(res["rec_quality"].sum()))
    print("q fraction of bins (all records)", round(float(res["q"].mean()), 3),
          "cov fraction", round(float(res["cov"].mean()), 3))
    for k in ("frame_acc", "frame_dec", "frame_uc"):
        print(k, "positive bins", int(res[k].sum()))
    print("dtype counts", np.unique(res["frame_dtype"][res["frame_dtype"] >= 0], return_counts=True))
    print("events", len(res["ev_rec"]), "by label", np.bincount(res["ev_label"]))
    wm, wn = res["win_masked"], res["win_nomask"]
    for j, b in enumerate(EXPERT_BITS):
        print(f"{b:15s} masked pos/neg/unk {int((wm[:, j] == 1).sum())}/{int((wm[:, j] == 0).sum())}/{int((wm[:, j] == -1).sum())}"
              f"  nomask pos {int((wn[:, j] == 1).sum())}")
