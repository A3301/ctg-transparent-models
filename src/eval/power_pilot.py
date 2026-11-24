"""Pilot power check for planning the end-to-end model analysis (docs/analiz_plani_v2.md; planning only;
uses out-of-fold predictions of the benchmark models).

Question: does a continuous-pH endpoint narrow the uncertainty of model comparisons compared
with AUROC for pH < 7.05 (38 positives)? Both metrics are on the same 0.5-1 scale:
  * AUROC: P(risk_pos > risk_neg) over positive/negative pairs (pH < 7.05).
  * Harrell's C for continuous pH: P(risk_i > risk_j | pH_i < pH_j) over all pairs with
    different pH (ties in risk count 1/2).
For each benchmark model: repeat-averaged OOF risk, metric, record bootstrap 95 % CI; for model
pairs: paired-bootstrap CI of the difference. The benchmark models were trained for pH < 7.05, not
for continuous pH, so C values are not results of this paper; only CI widths matter here.

Run: .venv\\Scripts\\python -m src.eval.power_pilot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.ctu_uhb import load_metadata  # noqa: E402

MODELS = ["cnn_ens_W60", "nesy_ens_W60", "transformer_W60", "figo_lr_W60"]
SEED, N_BOOT = 20261123, 1000


def oof(exp):
    acc = {}
    for f in sorted((ROOT / "results" / "cv" / exp).glob("r*_f*.npz")):
        d = np.load(f, allow_pickle=True)  # our own benchmark outputs
        for r, p in zip(d["records_te"], d["p_te"]):
            acc.setdefault(str(r), []).append(float(p))
    return pd.Series({r: np.mean(v) for r, v in acc.items()})


def cindex(risk, ph):
    """Harrell's C: higher risk should go with lower pH. O(n^2) vectorised; n ~ 536."""
    dr = risk[:, None] - risk[None, :]
    dp = ph[None, :] - ph[:, None]          # >0 when ph_i < ph_j
    comp = dp > 0
    conc = (dr > 0) & comp
    tie = (dr == 0) & comp
    return (conc.sum() + 0.5 * tie.sum()) / comp.sum()


def auroc(risk, y):
    pos, neg = risk[y == 1], risk[y == 0]
    d = pos[:, None] - neg[None, :]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size


def main():
    meta = load_metadata()
    meta.index = meta.index.astype(str)
    P = pd.DataFrame({m: oof(m) for m in MODELS}).dropna()
    ph = meta.loc[P.index, "ph"].astype(float)
    ok = ph.notna()
    P, ph = P[ok], ph[ok].values
    y = (ph < 7.05).astype(int)
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(ph), len(ph)) for _ in range(N_BOOT)]
    res = {"n": int(len(ph)), "n_pos_ph705": int(y.sum()), "models": {}, "differences": {}}
    boots = {}
    for m in MODELS:
        r = P[m].values
        a, c = auroc(r, y), cindex(r, ph)
        ba = np.array([auroc(r[b], y[b]) for b in idx])
        bc = np.array([cindex(r[b], ph[b]) for b in idx])
        boots[m] = (ba, bc)
        res["models"][m] = {"auroc": [round(a, 4), *np.round(np.percentile(ba, [2.5, 97.5]), 4).tolist()],
                            "auroc_ci_width": round(float(np.diff(np.percentile(ba, [2.5, 97.5]))[0]), 4),
                            "cindex_ph": [round(c, 4), *np.round(np.percentile(bc, [2.5, 97.5]), 4).tolist()],
                            "cindex_ci_width": round(float(np.diff(np.percentile(bc, [2.5, 97.5]))[0]), 4)}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            da = boots[m1][0] - boots[m2][0]
            dc = boots[m1][1] - boots[m2][1]
            res["differences"][f"{m1} - {m2}"] = {
                "d_auroc_ci": np.round(np.percentile(da, [2.5, 97.5]), 4).tolist(),
                "d_auroc_se": round(float(da.std()), 4),
                "d_cindex_ci": np.round(np.percentile(dc, [2.5, 97.5]), 4).tolist(),
                "d_cindex_se": round(float(dc.std()), 4),
                "se_ratio_auroc_over_c": round(float(da.std() / dc.std()), 2)}
    out = ROOT / "results" / "annotations" / "power_pilot.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
