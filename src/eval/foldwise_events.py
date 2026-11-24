"""Event-level fidelity with fold-specific thresholds (deviation log #15).

The pre-specified evaluation applied the median of the 50 fold thresholds to repeat-averaged frame probabilities.
Because a record is in the inner validation split of other outer folds, that median can depend indirectly on the
record's own expert labels. Here every outer-test record is thresholded only with the threshold selected on its own
fold's inner validation split; event counts are pooled over records within each repeat, F1 is computed per repeat
and averaged over the 5 repeats. Uncertainty: paired bootstrap over records (same resampled records for all sources
and repeats), 2000 resamples, as in the pre-specified analysis. The pre-registered comparisons are recomputed with
this protocol and adjusted with the same Holm families:
  - Plan 1 (v2):  H1a (DEC), H1b (ACC) NeSy-E2E vs extractor, with H1c (window concept, unaffected by thresholds);
  - Plan 2 (v3):  C2 (additive vs FIGO-feature model, from study3_results.json), B1a (DEC), B1b (ACC) sequential
                  detector vs extractor;
  - Addendum 2:   E1a (DEC), E1b (ACC) rule-guided detector vs extractor.
The rule-based extractor has no threshold, so its counts are identical in every repeat.

Run after src.eval.study3_eval (for C2). Output: results/study3/foldwise_events.json (aggregate only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.concept_targets import load_targets  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.concept_fidelity import detector, match  # noqa: E402
from src.eval.events import MIN_BINS, bins_to_events, expert_events, in_region  # noqa: E402
from src.eval.study2_eval import N_BOOT, SEED, load_nesy2  # noqa: E402

LAB = {0: "ACC", 1: "DEC", 2: "UC"}
SOURCES = {"nesy2": "nesy2_W60", "det": "det_W60", "detk": "detk_W60"}


def f1_from_sums(s):
    """s[..., 3] = (tp, n_ref, n_det) summed over records -> F1."""
    den = s[..., 1] + s[..., 2]
    return np.where(den > 0, 2 * s[..., 0] / np.maximum(den, 1), np.nan)


def main():
    T = load_targets()
    recs = [str(r) for r in T["records"]]
    pos = {r: i for i, r in enumerate(recs)}
    z = np.load(ROOT / "data" / "processed" / "windows_W60.npz")
    loaded = {k: load_nesy2(v) for k, v in SOURCES.items()}
    test_recs = sorted(set(loaded["nesy2"]["df"].record))
    assert len(test_recs) == 536 and all(set(loaded[k]["df"].record) == set(test_recs) for k in loaded)
    n = len(test_recs)

    rule_ev = {}
    for r in test_recs:
        i = pos[r]
        _, accs, decs, cons = detector(z["fhr"][i], z["fhr_mask"][i], z["uc"][i], z["uc_mask"][i])
        rule_ev[r] = {0: [(d["start"], d["end"]) for d in accs], 1: [(d["start"], d["end"]) for d in decs],
                      2: [(c["start"], c["end"]) for c in cons]}

    # counts[source][iou][lab] -> (5 repeats, n records, 3)
    counts = {}
    for iou in (0.0, 0.5):
        for lab in (0, 1, 2):
            ref = {}
            for r in test_recs:
                i = pos[r]
                reg = T["q"][i].astype(float)
                ref[r] = (reg, in_region(expert_events(T, i, lab), reg))
            c_rule = np.array([[len(match(ref[r][1], in_region(rule_ev[r][lab], ref[r][0]), iou)), len(ref[r][1]),
                                len(in_region(rule_ev[r][lab], ref[r][0]))] for r in test_recs], float)
            counts.setdefault("rule", {}).setdefault(iou, {})[lab] = np.repeat(c_rule[None], 5, axis=0)
            for src, L in loaded.items():
                arr = np.zeros((5, n, 3))
                for k, r in enumerate(test_recs):
                    reg, rf = ref[r]
                    for rep in range(5):
                        fr, th = L["per_rep"][(r, rep)]
                        det = in_region(bins_to_events(fr[lab], th[lab], MIN_BINS[lab]), reg)
                        arr[rep, k] = (len(match(rf, det, iou)), len(rf), len(det))
                counts.setdefault(src, {}).setdefault(iou, {})[lab] = arr

    rng = np.random.default_rng(SEED)
    W = np.zeros((N_BOOT, n))
    for b in range(N_BOOT):
        np.add.at(W[b], rng.integers(0, n, n), 1)

    def point(c):                      # mean over repeats of pooled F1, Se, PPV
        s = c.sum(1)                   # (5, 3)
        f1 = f1_from_sums(s)
        return dict(f1=float(np.mean(f1)), se=float(np.mean(s[:, 0] / s[:, 1])), ppv=float(np.mean(s[:, 0] / np.maximum(s[:, 2], 1))),
                    f1_per_repeat=[float(v) for v in f1])

    def boot(c):                       # (N_BOOT,) mean-over-repeat F1
        s = np.einsum("bn,rnk->brk", W, c)
        return np.nanmean(f1_from_sums(s), axis=1)

    out = {"protocol": __doc__.split("\n\n")[0], "n_records": n, "n_boot": N_BOOT, "events": {}, "tests": {}}
    B = {}
    for src in ("nesy2", "det", "detk", "rule"):
        for iou in (0.0, 0.5):
            for lab in (0, 1, 2):
                c = counts[src][iou][lab]
                key = f"{src}{'_iou50' if iou else ''}|{LAB[lab]}"
                bb = boot(c)
                B[key] = bb
                out["events"][key] = {**point(c), "f1_ci": [point(c)["f1"], float(np.nanquantile(bb, .025)), float(np.nanquantile(bb, .975))]}

    def diff(a, b):
        d = B[a] - B[b]
        obs = out["events"][a]["f1"] - out["events"][b]["f1"]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return dict(diff=float(obs), lo=float(np.nanquantile(d, .025)), hi=float(np.nanquantile(d, .975)), p=float(min(p, 1)))

    t = out["tests"]
    t["H1a_DEC"], t["H1b_ACC"] = diff("nesy2|DEC", "rule|DEC"), diff("nesy2|ACC", "rule|ACC")
    t["B1a_DEC"], t["B1b_ACC"] = diff("det|DEC", "rule|DEC"), diff("det|ACC", "rule|ACC")
    t["E1a_DEC"], t["E1b_ACC"] = diff("detk|DEC", "rule|DEC"), diff("detk|ACC", "rule|ACC")
    for lab in ("ACC", "DEC", "UC"):     # descriptive
        t[f"detk_minus_det|{lab}"] = diff(f"detk|{lab}", f"det|{lab}")
        t[f"detk_minus_rule_iou50|{lab}"] = diff(f"detk_iou50|{lab}", f"rule_iou50|{lab}")
        t[f"det_minus_nesy2|{lab}"] = diff(f"det|{lab}", f"nesy2|{lab}")
    for src in ("nesy2", "det", "detk"):  # contractions were not part of the pre-registered event tests
        t[f"{src}_minus_rule|UC"] = diff(f"{src}|UC", "rule|UC")

    S2 = json.loads((ROOT / "results/study2/study2_results.json").read_text(encoding="utf-8"))
    S3 = json.loads((ROOT / "results/study3/study3_results.json").read_text(encoding="utf-8"))
    fam = {"plan1": (["H1a_DEC", "H1b_ACC"], S2["H1"]["H1c_accel_window"]["p"]),
           "plan2": (["B1a_DEC", "B1b_ACC"], S3["superiority"]["C2"]["p"]),
           "addendum2": (["E1a_DEC", "E1b_ACC"], None)}
    for name, (keys, extra_p) in fam.items():
        ps = [t[k]["p"] for k in keys] + ([extra_p] if extra_p is not None else [])
        adj = M.holm(ps)
        for k, a in zip(keys, adj):
            t[k]["p_holm"] = float(a)
            t[k]["holm_family"] = name
            t[k]["supported"] = bool(t[k]["diff"] > 0 and a < 0.05)
    out["holm_family_extra_p"] = {"plan1_H1c": S2["H1"]["H1c_accel_window"]["p"], "plan2_C2": S3["superiority"]["C2"]["p"]}
    # Holm-adjusted p of the other members of each family under the fold-specific protocol (H1c, C2)
    adj1 = M.holm([t["H1a_DEC"]["p"], t["H1b_ACC"]["p"], S2["H1"]["H1c_accel_window"]["p"]])
    adj2 = M.holm([t["B1a_DEC"]["p"], t["B1b_ACC"]["p"], S3["superiority"]["C2"]["p"]])
    out["family_partners"] = {"H1c_p_holm": float(adj1[2]),
                              "C2": {**S3["superiority"]["C2"], "p_holm_foldwise_family": float(adj2[2]),
                                     "supported_foldwise_family": bool(S3["superiority"]["C2"]["diff"] > 0 and adj2[2] < 0.05)}}

    # ---- secondary, added after the primary fold-specific results were seen (deviation log #15, note):
    # repeat-averaged frame probabilities as in the pre-specified analysis, but each record thresholded with the median of
    # the 5 thresholds of ITS OWN test folds (chosen on validation splits that exclude the record) -> also leakage-free.
    sec = {}
    for src, L in loaded.items():
        for lab in (0, 1):
            arr = np.zeros((1, n, 3))
            for k, r in enumerate(test_recs):
                i = pos[r]
                reg = T["q"][i].astype(float)
                rf = in_region(expert_events(T, i, lab), reg)
                frs = [L["per_rep"][(r, rep)] for rep in range(5)]
                prob = np.mean([fr[lab] for fr, _ in frs], 0)
                thr = float(np.median([th[lab] for _, th in frs]))
                det = in_region(bins_to_events(prob, thr, MIN_BINS[lab]), reg)
                arr[0, k] = (len(match(rf, det)), len(rf), len(det))
            key = f"{src}|{LAB[lab]}"
            bb = np.nanmean(f1_from_sums(np.einsum("bn,rnk->brk", W, arr)), axis=1)
            sec[key] = dict(f1=float(f1_from_sums(arr.sum(1))[0]), boot=bb)
    for lab in ("ACC", "DEC"):
        c_rule = counts["rule"][0.0][{"ACC": 0, "DEC": 1}[lab]][:1]
        sec[f"rule|{lab}"] = dict(f1=float(f1_from_sums(c_rule.sum(1))[0]),
                                  boot=np.nanmean(f1_from_sums(np.einsum("bn,rnk->brk", W, c_rule)), axis=1))
    s_out = {k: {"f1": v["f1"], "f1_ci": [v["f1"], float(np.nanquantile(v["boot"], .025)), float(np.nanquantile(v["boot"], .975))]}
             for k, v in sec.items()}
    for a, b, name in (("detk", "rule", "E1"), ("det", "rule", "B1"), ("nesy2", "rule", "H1")):
        for lab in ("DEC", "ACC"):
            d = sec[f"{a}|{lab}"]["boot"] - sec[f"{b}|{lab}"]["boot"]
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            s_out[f"{name}_{lab}"] = dict(diff=sec[f"{a}|{lab}"]["f1"] - sec[f"{b}|{lab}"]["f1"], lo=float(np.nanquantile(d, .025)),
                                          hi=float(np.nanquantile(d, .975)), p=float(min(p, 1)))
    out["secondary_record_median_threshold"] = s_out
    for k in ("E1_DEC", "E1_ACC", "B1_DEC", "B1_ACC"):
        v = s_out[k]
        print(f"secondary {k}: {v['diff']:+.3f} ({v['lo']:+.3f} to {v['hi']:+.3f}) p {v['p']:.3f}")

    path = ROOT / "results/study3/foldwise_events.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    from src.utils.repro import log_experiment
    log_experiment("S3-fold", "foldwise_events", {"script": "src/eval/foldwise_events.py", "deviation": "#15", "n_boot": N_BOOT}, SEED,
                   {k: {kk: t[k][kk] for kk in ("diff", "lo", "hi", "p", "p_holm")} for k in ("H1a_DEC", "H1b_ACC", "B1a_DEC", "B1b_ACC", "E1a_DEC", "E1b_ACC")},
                   "event fidelity with fold-specific thresholds")
    for k in ("H1a_DEC", "H1b_ACC", "B1a_DEC", "B1b_ACC", "E1a_DEC", "E1b_ACC"):
        v = t[k]
        print(f"{k:8s} diff {v['diff']:+.3f} ({v['lo']:+.3f} to {v['hi']:+.3f}) p {v['p']:.3f} holm {v['p_holm']:.3f} supported {v['supported']}")
    for k in ("nesy2|ACC", "nesy2|DEC", "det|ACC", "det|DEC", "detk|ACC", "detk|DEC", "rule|ACC", "rule|DEC", "detk_iou50|ACC", "rule_iou50|ACC"):
        print(k, round(out["events"][k]["f1"], 3), [round(x, 3) for x in out["events"][k]["f1_ci"][1:]])


if __name__ == "__main__":
    main()
