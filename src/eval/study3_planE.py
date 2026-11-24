"""Plan E evaluation (docs/analiz_plani_v3_ek2.md), written before any Plan E result was seen.

Run: python -m src.eval.study3_planE  (with the project .venv)
Output: results/study3/planE_results.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.study2_eval import N_BOOT, SEED, boot_f1_diff, repeat_avg  # noqa: E402
from src.eval.study3_eval import detector_fidelity, fhrma_detector, per_repeat_auroc  # noqa: E402


def main():
    for e in ("detk_W60", "transpdetk_gam_W60", "det_W60"):
        assert len(glob.glob(str(ROOT / "results" / "cv" / e / "r*_f*.npz"))) == 50, e
    rng = np.random.default_rng(SEED)
    T = load_targets()
    pos = {r: i for i, r in enumerate(str(x) for x in T["records"])}
    z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
    cnn = repeat_avg("cnn_ens_W60")
    test_recs = sorted(cnn.index)
    res = {}
    ev, e1, thr = detector_fidelity(T, test_recs, pos, z, rng, exp="detk_W60")
    res["detk_events"], res["detk_thresholds_median"] = ev, thr.tolist()
    keys = ["B1a_DEC", "B1b_ACC"]
    adj = M.holm([e1[k]["p"] for k in keys])
    res["E1"] = {k.replace("B1", "E1"): {**e1[k], "p_holm": float(a), "supported": bool(e1[k]["diff"] > 0 and a < 0.05)}
                 for k, a in zip(keys, adj)}
    # E4: detk vs det (descriptive): rebuild per-record counts for both on the same region
    from src.eval.concept_fidelity import match
    from src.eval.events import MIN_BINS, bins_to_events, expert_events, in_region
    from src.eval.study2_eval import load_nesy2
    nk, nd = load_nesy2("detk_W60"), load_nesy2("det_W60")
    e4 = {}
    for lab, nm in ((0, "ACC"), (1, "DEC"), (2, "UC")):
        ca, cb = [], []
        for r in test_recs:
            i = pos[r]
            reg = T["q"][i].astype(float)
            ref = in_region(expert_events(T, i, lab), reg)
            for nz, c in ((nk, ca), (nd, cb)):
                det = in_region(bins_to_events(nz["frame"][r][lab], nz["thr"][lab], MIN_BINS[lab]), reg)
                c.append([len(match(ref, det)), len(ref), len(det)])
        e4[nm] = boot_f1_diff(np.array(ca), np.array(cb), rng)
    res["E4_detk_minus_det"] = e4
    try:
        res["E2_fhrma"] = fhrma_detector(thr, exp="detk_W60")
    except Exception as ex:  # noqa: BLE001
        res["E2_fhrma"] = f"failed: {ex!r}"
    y = cnn.loc[test_recs].y.values
    p = repeat_avg("transpdetk_gam_W60").loc[test_recs].p.values
    d = M.paired_bootstrap_diff(y, p, cnn.loc[test_recs].p.values, M.auroc, n=N_BOOT, seed=SEED)
    res["E3_transpdetk_gam"] = dict(auroc=M.bootstrap_ci(y, p, M.auroc, n=N_BOOT, seed=SEED),
                                    auprc=M.bootstrap_ci(y, p, M.auprc, n=N_BOOT, seed=SEED),
                                    brier=M.bootstrap_ci(y, p, M.brier, n=N_BOOT, seed=SEED),
                                    diff_vs_cnn=dict(diff=d[0], lo=d[1], hi=d[2], p=d[3]),
                                    per_repeat=per_repeat_auroc("transpdetk_gam_W60"))
    out = ROOT / "results" / "study3" / "planE_results.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("E1", "E4_detk_minus_det", "E3_transpdetk_gam")}, indent=1, default=float))


if __name__ == "__main__":
    main()
