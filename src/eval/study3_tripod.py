"""TRIPOD+AI items for the transparent models (plan dışı #12, descriptive): cohort characteristics (item 20b),
participant flow and numbers per analysis (20a, 21), subgroup performance (23a) and the full
specification of the final linear transparent model (22).

Subgroups use only unambiguously coded metadata (parity, maternal age, gestational age, induction,
delivery type; deliv_type 2 = caesarean matches the 46 caesareans stated by PhysioNet). Acquisition type
and fetal sex are not used because their numeric codes could not be verified.
Metadata are used for description only, never as model inputs.

Run: python -m src.eval.study3_tripod   (with the project .venv)
Output: results/study3/tripod_results.json, results/study3/transp_lr_final_model.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval import metrics as M  # noqa: E402
from src.eval.study2_eval import N_BOOT, SEED, repeat_avg  # noqa: E402


def med_iqr(x, d=1):
    x = pd.Series(x).dropna()
    return f"{x.median():.{d}f} ({x.quantile(.25):.{d}f}–{x.quantile(.75):.{d}f})"


def pct(b):
    b = pd.Series(b).dropna().astype(bool)
    return f"{int(b.sum())} ({100 * b.mean():.1f}%)"


def main():
    meta = pd.read_csv(ROOT / "results/eda/metadata_all.csv", dtype={"record": str}).set_index("record")
    wq = pd.read_csv(ROOT / "results/preprocess/window_quality.csv", dtype={"record": str})
    wq = wq[wq.window == "W60"].set_index("record")
    cnn = repeat_avg("cnn_ens_W60")
    recs = sorted(cnn.index)
    y = cnn.loc[recs].y.values.astype(int)
    m = meta.loc[recs]
    out = {"flow": {"database": int(len(meta)), "excluded_fhr_loss_gt50": int(len(meta) - len(recs)), "analysed": len(recs),
                    "events_ph_lt_7_05": int(y.sum())}}
    # ---- 20b characteristics by outcome
    rows = {}
    for name, grp in (("all", np.ones(len(recs), bool)), ("ph_lt_7_05", y == 1), ("ph_ge_7_05", y == 0)):
        g = m[grp]
        rows[name] = {"n": int(grp.sum()), "maternal_age_y": med_iqr(g.mat_age), "nulliparous": pct(g.parity == 0),
                      "gestational_age_wk": med_iqr(g.gest_weeks), "gest_ge_41wk": pct(g.gest_weeks >= 41),
                      "induced_labour": pct(g.induced == 1), "caesarean": pct(g.deliv_type == 2),
                      "meconium": pct(g.meconium == 1), "birth_weight_g": med_iqr(g.weight_g),
                      "stage2_min_vaginal": med_iqr(g.stage2_min.where(g.stage2_min >= 0)), "ph": med_iqr(g.ph, 2), "bdecf_mmol_l": med_iqr(g.bdecf),
                      "apgar5_lt7": pct(g.apgar5 < 7),
                      "fhr_missing_W60_pct": med_iqr(100 * wq.loc[g.index, "fhr_missing"]),
                      "uc_missing_W60_pct": med_iqr(100 * wq.loc[g.index, "uc_missing"])}
    out["characteristics"] = rows
    # ---- 21 numbers per analysis (outer folds)
    splits = json.loads((ROOT / "results/splits/outer_splits.json").read_text())
    yd = dict(zip(recs, y))
    tr_n = [len(f["train"]) for rep in splits["repeats"] for f in rep]
    te_n = [len(f["test"]) for rep in splits["repeats"] for f in rep]
    tr_e = [sum(yd[r] for r in f["train"]) for rep in splits["repeats"] for f in rep]
    te_e = [sum(yd[r] for r in f["test"]) for rep in splits["repeats"] for f in rep]
    out["per_fold"] = dict(train_n=[min(tr_n), max(tr_n)], train_events=[min(tr_e), max(tr_e)], test_n=[min(te_n), max(te_n)],
                           test_events=[min(te_e), max(te_e)], fit_fraction=0.85, inner_folds=3)
    # ---- 23a subgroups
    P = {k: repeat_avg(v).loc[recs].p.values for k, v in
         (("cnn_bayes", "cnn_ens_W60"), ("transp_lr", "transp_lr_W60"), ("transp_gam", "transp_gam_W60"))}
    subs = {"nulliparous": m.parity.values == 0, "parous": m.parity.values > 0,
            "maternal_age_lt35": m.mat_age.values < 35, "maternal_age_ge35": m.mat_age.values >= 35,
            "gest_lt41": m.gest_weeks.values < 41, "gest_ge41": m.gest_weeks.values >= 41,
            "induced": m.induced.values == 1, "spontaneous": m.induced.values == 0,
            "vaginal": m.deliv_type.values == 1, "caesarean": m.deliv_type.values == 2}
    sg = {}
    for s, mask in subs.items():
        ys = y[mask]
        sg[s] = {"n": int(mask.sum()), "events": int(ys.sum())}
        for k, p in P.items():
            sg[s][k] = (M.bootstrap_ci(ys, p[mask], M.auroc, n=N_BOOT, seed=SEED) if 0 < ys.sum() < len(ys) and ys.sum() >= 5
                        else "too few events")
    out["subgroups"] = sg
    # ---- 22 final model specification: transp_lr refit on all 536 with the modal selected C
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.run.cv import transp_table
    t = transp_table("transp_lr", "W60").loc[recs]
    X = t.values.astype(float)
    cs = [json.loads(str(np.load(fp, allow_pickle=True)["cfg"]))["C"] for fp in sorted((ROOT / "results/cv/transp_lr_W60").glob("r*_f*.npz"))]
    C = float(np.median(cs))
    imp, sc = SimpleImputer(strategy="median").fit(X), None
    Xi = imp.transform(X)
    sc = StandardScaler().fit(Xi)
    Z = sc.transform(Xi)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z, y)
    pw = (len(y) - y.sum()) / y.sum()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        boots.append(LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z[idx], y[idx]).coef_[0])
    boots = np.array(boots)
    coef = [dict(variable=c, median_imputation=float(imp.statistics_[j]), mean=float(sc.mean_[j]), sd=float(sc.scale_[j]),
                 coef_per_sd=float(lr.coef_[0][j]), coef_ci95=[float(np.quantile(boots[:, j], .025)), float(np.quantile(boots[:, j], .975))])
            for j, c in enumerate(t.columns)]
    spec = {"model": "L2 logistic regression on standardised variables (class_weight=balanced)", "C": C,
            "C_selected_per_fold_median_of": len(cs), "intercept_balanced": float(lr.intercept_[0]),
            "prior_correction": f"subtract log(pos_weight) = {np.log(pw):.4f} from the linear predictor",
            "formula": "logit(p) = intercept_balanced - log(pos_weight) + sum_j coef_per_sd_j * (x_j_imputed - mean_j) / sd_j",
            "note": "Descriptive specification refitted on all 536 recordings; temperature scaling (fold-specific) not included; "
                    "not externally validated.", "coefficients": coef}
    (ROOT / "results/study3/transp_lr_final_model.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    out["final_model_top_coefficients"] = sorted([(c["variable"], round(c["coef_per_sd"], 3), [round(v, 3) for v in c["coef_ci95"]])
                                                  for c in coef], key=lambda r: -abs(r[1]))[:10]
    out["selected_C_distribution"] = dict(Counter(np.round(np.log10(cs), 1)).most_common(5))
    (ROOT / "results/study3/tripod_results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("flow", "per_fold", "final_model_top_coefficients")}, indent=1, default=str))
    for s, v in sg.items():
        print(s, v["n"], v["events"], {k: (round(v[k][0], 3) if isinstance(v[k], tuple) or isinstance(v[k], list) else v[k]) for k in P})
    for k, v in rows.items():
        print(k, v)


if __name__ == "__main__":
    main()
