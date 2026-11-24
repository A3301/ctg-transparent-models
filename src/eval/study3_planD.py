"""Plan D (docs/analiz_plani_v3_ek1.md): transparent models vs the benchmark CNN ensemble and FIGO-LR across the
benchmark sensitivity settings. Descriptive only (no tests, no Holm).

Run: .venv\\Scripts\\python -m src.eval.study3_planD
Output: results/study3/planD_results.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402
from src.eval.study2_eval import N_BOOT, SEED, repeat_avg  # noqa: E402

SETTINGS = ["W60", "ph715", "bdecf12", "apgar5", "all552", "W30", "W90", "S1_60"]


def exp(model, s):
    if s == "W60":
        return {"cnn": "cnn_ens_W60", "figo": "figo_lr_W60"}.get(model, f"{model}_W60")
    return {"cnn": f"cnn_ens_{s}", "figo": f"figo_lr_{s}"}.get(model, f"{model}_{s}")


def per_repeat(e):
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / e / "r*_f*.npz"))):
        z = np.load(fp, allow_pickle=True)
        r = int(Path(fp).stem[1:].split("_f")[0])
        rows += [dict(repeat=r, y=int(a), p=float(b)) for a, b in zip(z["y_te"], z["p_te"])]
    d = pd.DataFrame(rows)
    v = d.groupby("repeat").apply(lambda g: M.auroc(g.y.values, g.p.values))
    return float(v.mean()), float(v.std())


def main():
    out = {}
    for s in SETTINGS:
        P = {m: repeat_avg(exp(m, s)) for m in ("cnn", "figo", "transp_lr", "transp_gam")}
        for m in P:
            assert len(glob.glob(str(ROOT / "results" / "cv" / exp(m, s) / "r*_f*.npz"))) == 50, (m, s)
        recs = sorted(P["cnn"].index)
        y = P["cnn"].loc[recs].y.values.astype(int)
        for m in P:
            assert P[m].loc[recs].y.values.astype(int).tolist() == y.tolist(), (m, s)
        row = {"n": len(recs), "n_pos": int(y.sum())}
        for m in P:
            p = P[m].loc[recs].p.values
            row[m] = dict(auroc=M.bootstrap_ci(y, p, M.auroc, n=N_BOOT, seed=SEED),
                          auprc=M.auprc(y, p), per_repeat=per_repeat(exp(m, s)))
        for m in ("transp_lr", "transp_gam"):
            row[f"{m}_minus_cnn"] = M.paired_bootstrap_diff(y, P[m].loc[recs].p.values, P["cnn"].loc[recs].p.values,
                                                           M.auroc, n=N_BOOT, seed=SEED)
            row[f"{m}_minus_figo"] = M.paired_bootstrap_diff(y, P[m].loc[recs].p.values, P["figo"].loc[recs].p.values,
                                                            M.auroc, n=N_BOOT, seed=SEED)
        out[s] = row
    summ = {m: dict(n_settings_diff_gt0_vs_cnn=int(sum(out[s][f"{m}_minus_cnn"][0] > 0 for s in SETTINGS[1:])),
                    n_settings_lo_gt_minus005_vs_cnn=int(sum(out[s][f"{m}_minus_cnn"][1] > -0.05 for s in SETTINGS[1:])),
                    median_diff_vs_cnn=float(np.median([out[s][f"{m}_minus_cnn"][0] for s in SETTINGS[1:]])))
            for m in ("transp_lr", "transp_gam")}
    out["summary_over_7_sensitivity_settings"] = summ
    p = ROOT / "results" / "study3" / "planD_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for s in SETTINGS:
        r = out[s]
        print(f"{s:8s} n={r['n']} pos={r['n_pos']:3d} | cnn {r['cnn']['auroc'][0]:.3f} figo {r['figo']['auroc'][0]:.3f} "
              f"lr {r['transp_lr']['auroc'][0]:.3f} gam {r['transp_gam']['auroc'][0]:.3f} | "
              f"lr-cnn {r['transp_lr_minus_cnn'][0]:+.3f} [{r['transp_lr_minus_cnn'][1]:+.3f},{r['transp_lr_minus_cnn'][2]:+.3f}] | "
              f"rep lr {r['transp_lr']['per_repeat'][0]:.3f} cnn {r['cnn']['per_repeat'][0]:.3f}")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
