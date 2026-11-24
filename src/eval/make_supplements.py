"""Build the supplementary material directly from the result files (no hand-typed numbers).

Sections are numbered in order of their first citation in the manuscript (src/eval/supp_numbering.py); the
manuscript cites them by key, e.g. "Supplementary S[prereg]".

Run: python -m src.eval.make_supplements   (with the project .venv)
Output: manuscript/combined/supplement_combined.md and two section subsets (manuscript/supplement_paper1.md,
manuscript/supplement_paper2.md). The section numbering reads the manuscript sources, which are not part of the
public code release; the released supplement is manuscript/supplementary_material.md.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.supp_numbering import KEYS, numbers, resolve  # noqa: E402
from src.eval.explain import PRETTY, RULE_TXT  # noqa: E402

R = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731

PLAN_ROLE = {
    "docs/analiz_plani_v3.md": "Plan 2 (sequential detector; transparent models)",
    "docs/analiz_plani_v3_ek1.md": "Addendum 1 (robustness settings)",
    "docs/analiz_plani_v3_ek2.md": "Addendum 2 (rule-guided detector)",
    "docs/analiz_plani_v3_ek3.md": "Addendum 3 (distribution shifts)",
    "docs/analiz_plani_v3_ek4.md": "Addendum 4 (distribution-shift fallbacks)",
}
PLANS_USED = {"combined": ["docs/analiz_plani_v3.md", "docs/analiz_plani_v3_ek1.md", "docs/analiz_plani_v3_ek2.md",
                           "docs/analiz_plani_v3_ek3.md", "docs/analiz_plani_v3_ek4.md"],
              "paper1": ["docs/analiz_plani_v3.md", "docs/analiz_plani_v3_ek2.md"],
              "paper2": ["docs/analiz_plani_v3.md", "docs/analiz_plani_v3_ek1.md", "docs/analiz_plani_v3_ek3.md",
                         "docs/analiz_plani_v3_ek4.md"]}

# Deviations from the frozen plans (docs/plan_disi_degisiklikler.md), as they apply to each section subset
DEVIATIONS = {
    "paper1": [
        ("8", "Plan 1, hypothesis H1c", "Exploratory diagnostic check of the chance-level acceleration concept (index and record "
         "alignment, per-fold AUROC). No model, threshold or primary result was changed.", "After the Plan 1 results"),
        ("9", "Plan 1, interpretation rule of the intervention test", "The rule (a small change means that the outcome flows "
         "through the concepts) assumed concepts close to the expert reference. With concepts near chance, a small change was "
         "not interpreted as evidence that the risk flows through the expert-referenced concepts. Numbers unchanged.", "After the Plan 1 results"),
        ("10", "Plan 2, tuning budget (6 random configurations)", "The additive risk models (on detector concepts and, as comparator, "
         "on rule-extractor features) were tuned with 30 configurations, as in the benchmark protocol released with the code; 6 applies to the deep detectors.",
         "While writing the code, before any result of this study"),
        ("13", "Code (rule-based extractor)", "Correction of a coding error: when overlapping decelerations were merged, their areas were "
         "summed, so overlapping samples were counted twice. The merged event's area is now computed once over the merged span. Every model "
         "using the FIGO variables was re-run with the same protocol; detector, NeSy and CNN outputs were unaffected.", "After all results"),
        ("15", "Plans 1 and 2, event threshold for outer-test records (median of the 50 fold thresholds applied to repeat-averaged "
         "probabilities)", "Main analysis changed to fold-specific thresholds: each outer-test record is thresholded with its own fold's "
         "validation-selected threshold; F1 per repeat, averaged over repeats; pre-specified comparisons recomputed with the same bootstrap and "
         "Holm families. The pooled median threshold depends indirectly on a record's own labels through other folds' validation splits. "
         "The choice of the main analysis was logged before its results were seen; the pre-specified results are reported alongside.",
         "After the pre-specified results; before the fold-specific results"),
        ("17", "Plan 1, intervention test", "Added a check of how well the approximate rebuild reproduces the recorded risks.", "After all results"),
        ("18", "Plans 1 and 2, event threshold", "Added a secondary descriptive variant: each record is thresholded with the median of the "
         "thresholds of the folds in which it was an outer-test record (no test record's labels enter its own threshold).",
         "After the fold-specific results"),
        ("19", "Scope (post hoc)", "Added member-level concept substitution and a controlled outcome-loss ablation. "
         "The original analyses are reported alongside.", "After the original results"),
    ],
    "paper2": [
        ("10", "Analysis plan, tuning budget (6 random configurations)", "The scikit-learn models (transparent models and the FIGO-feature "
         "model) were tuned with 30 configurations, as in the benchmark protocol released with the code; 6 applies to the deep networks.",
         "While writing the code, before any result of this study"),
        ("11", "Analysis plan, list of measures", "Added descriptive reporting: sensitivity at 85% and 90% specificity, calibration slope and "
         "intercept, decision-curve net benefit and the ΔAUROC of each transparent model against the ensemble. No tests.",
         "After the primary results"),
        ("12", "Analysis plan, scope", "Added TRIPOD+AI descriptive items: cohort characteristics, numbers per analysis, subgroup AUROCs "
         "and the full specification of the refitted linear model. Clinical metadata were used for description only.",
         "After all results"),
        ("13", "Code (rule-based extractor)", "Correction of a coding error: when overlapping decelerations were merged, their areas were "
         "summed, so overlapping samples were counted twice. The merged event's area is now computed once over the merged span. Every model "
         "using the FIGO variables was re-run with the same protocol; detector, NeSy and CNN outputs were unaffected.", "After all results"),
        ("14", "Addendum 1, outcome \"pH < 7.15\"", "The label actually used was the benchmark's secondary outcome pH ≤ 7.15 (106 events); the "
         "text of the addendum contained a typing error. Reported as run; not re-run with < 7.15.", "After all results"),
        ("16", "Analysis plan, scope", "Added descriptive analyses: a linear model of the three HRV variables, HRV and signal-quality ablations of the "
         "linear model, and bootstrap CIs for the calibration slope and intercept. No tests.", "After all results"),
        ("18", "Analysis plan, complementarity", "Added the average of the linear model and the CNN ensemble alongside the pre-specified "
         "average of the additive model and the ensemble. Descriptive.", "After all results"),
        ("20", "Scope (post hoc)", "Added validation-selected operating points, a single-feature FHR signal loss comparator, "
         "and paired net-benefit intervals on the original out-of-fold predictions.", "After the original results"),
    ],
}


def _combined_deviations():
    """Deviations of this paper: the union of both section subsets, with the shared rows (#10, #18) merged."""
    p1, p2 = dict((r[0], r) for r in DEVIATIONS["paper1"]), dict((r[0], r) for r in DEVIATIONS["paper2"])
    merged = {**p2, **p1}
    merged["10"] = ("10", "Plan 2, tuning budget (6 random configurations)", "The scikit-learn models (additive and linear models, the "
                    "FIGO-feature model, the HRV models and the additive models on detector concepts) were tuned with 30 configurations, as in "
                    "the benchmark protocol released with the code; 6 applies to the deep networks.", "While writing the code, before any result of this study")
    merged["18"] = ("18", "Plans 1 and 2, event threshold and complementarity", "Added two descriptive analyses: a record-specific "
                    "threshold (the median of the thresholds of the folds in which the record was an outer-test record) and the average of the "
                    "linear model and the CNN ensemble alongside the pre-specified average of the additive model and the ensemble.",
                    "After the fold-specific and primary results")
    return [(a, b.replace("Analysis plan, ", "Plan 2, "), c, d) for a, b, c, d in (merged[k] for k in sorted(merged, key=int))]


DEVIATIONS["combined"] = _combined_deviations()


FIGO_MAP = [  # operationalisation of the FIGO 2015 criteria by the rule-based extractor (src/features/figo.py)
    ("Baseline", "Mean FHR level of the most horizontal, least oscillatory segments, estimated over 10-min periods",
     "Iterative robust mean within ±5 min of each sample (values > 10 bpm from the running estimate excluded, two passes), evaluated every 10 s and interpolated",
     "No visual selection of stable segments"),
    ("Variability", "Bandwidth amplitude of the FHR in 1-min segments; reduced < 5 bpm, increased > 25 bpm",
     "Per-minute P95–P5 bandwidth after removing accelerations and decelerations; median over the window", "P95–P5 replaces the visual peak-to-trough amplitude"),
    ("Reduced variability, pathological", "< 5 bpm for > 50 min (or > 3 min during decelerations)",
     "Minutes with bandwidth < 5 bpm summed over the window (not necessarily continuous); > 50 min in the 60-min window",
     "The criterion during decelerations is not implemented; in the 30- and 90-min sensitivity windows the duration is scaled with the window length"),
    ("Increased (saltatory) variability", "> 25 bpm for > 30 min", "Minutes with bandwidth > 25 bpm summed; > 30 min in the 60-min window", "As above"),
    ("Acceleration", "Abrupt increase > 15 bpm lasting > 15 s and < 10 min",
     "Excursion > 15 bpm above the baseline for > 15 s and ≤ 10 min, extended to the baseline crossings", "Abrupt onset is not checked"),
    ("Deceleration", "Decrease > 15 bpm lasting > 15 s",
     "Excursion > 15 bpm below the baseline for > 15 s, extended to the baseline crossings; overlapping events are merged and measured once over the merged span", ""),
    ("Early deceleration", "Shallow, short, coincident with the contraction", "Gradual (onset to nadir ≥ 30 s) with the nadir within ±20 s of a contraction peak",
     "Depth and within-deceleration variability are not used"),
    ("Late deceleration", "Gradual onset and return, nadir > 20 s after the contraction peak", "Gradual, with the nadir 20–60 s after the nearest contraction peak", ""),
    ("Variable deceleration", "Rapid drop (onset to nadir < 30 s)", "Onset to nadir < 30 s", "Within-deceleration variability is not used"),
    ("Prolonged deceleration", "> 3 min (pathological if > 5 min)", "Duration > 3 min; > 5 min flagged as pathological", ""),
    ("Repetitive decelerations", "Associated with > 50% of contractions",
     "Fraction of detected contractions with an associated deceleration (nadir within ±60 s of the peak) > 0.5", "Depends on contraction detection"),
    ("Unclassified deceleration", "Not a FIGO category",
     "Gradual deceleration not associated with a contraction peak, or without a valid contraction signal", "Introduced by the authors"),
    ("Sinusoidal pattern", "Smooth undulating pattern, 5–15 bpm, 3–5 cycles/min, > 30 min",
     "10-min segments with ≥ 50% of variability power at 3–5 cycles/min and amplitude 5–15 bpm; ≥ 30 min in the 60-min window", "Spectral approximation"),
    ("Contractions", "Tachysystole: > 5 contractions in 10 min",
     "Peaks of the 15-s smoothed UC signal (prominence ≥ 10 units, ≥ 60 s apart, half-width 15–180 s); contractions per 10 min", "Tachysystole is not used as a rule"),
]


def figo_section():
    b = ["The extractor is an operational approximation of FIGO 2015, not an executable form of the guideline. The table separates the "
         "guideline definition from the numerical rule used here.", "",
         "| Criterion | FIGO 2015 | Operationalisation in this study | Difference |", "|---|---|---|---|"]
    b += [f"| {a} | {g} | {o} | {d or '—'} |" for a, g, o, d in FIGO_MAP]
    return ("Operationalisation of the FIGO 2015 criteria", b)


def f3(x):
    return f"{x:.3f}"


def fp(p, floor="0.001"):
    """p-value; a bootstrap p of 0 is printed as below the resolution (2000 resamples)."""
    return f"< {floor}" if p == 0 else f"{p:.3f}"


def cap(s):
    return s[:1].upper() + s[1:]


def ci(v):
    return f"{v[0]:.3f} ({v[1]:.3f}–{v[2]:.3f})"


def plan_checksums(paper):
    rows = [r for r in csv.reader(open(ROOT / "experiments/log.csv", encoding="utf-8")) if len(r) > 2 and r[1] == "S3-plan"]
    out = ["| Plan file | Role | Frozen (UTC) | SHA-256 |", "|---|---|---|---|"]
    for r in rows:
        c = json.loads(r[3])
        if c["file"] in PLANS_USED[paper]:
            role = PLAN_ROLE.get(c['file'], '')
            if paper == "paper2" and c['file'] == "docs/analiz_plani_v3.md":
                role = "Main analysis plan (transparent models; also covers the sequential detector)"
            out.append(f"| `{c['file']}` | {role} | {r[0]} | `{c['sha256'][:16]}…` |")
    return out


def deviations(paper):
    out = ["**Deviations from the frozen plans** (recorded in `docs/plan_disi_degisiklikler.md`, with timing):", "",
           "| Log no. | Plan item | Deviation | Timing |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in DEVIATIONS[paper]]
    return out


def assemble(paper, title, sections):
    num = numbers(paper)
    assert set(sections) == set(KEYS[paper]), set(KEYS[paper]) ^ set(sections)
    L = ["# Supplementary material", "", f"*{title}* All tables are generated by `src/eval/make_supplements.py` from the result files.", ""]
    for k in sorted(sections, key=lambda k: num[k]):
        head, body = sections[k]
        L += [f"## S{num[k]}. {head}", ""] + body + [""]
    text = resolve("\n".join(L).rstrip() + "\n", paper)
    # typographic minus and unsigned CI bounds, as in the manuscripts (dates, 1e-3 and words are untouched)
    text = re.sub(r"(?<![\w.\-–/`])-(?=\d)", "−", text)
    text = re.sub(r"\(\+(?=\d)", "(", text)
    text = re.sub(r" to \+(?=\d)", " to ", text)
    return text


def paper1(sections_only=False):
    S2, S3, E, F2 = (R("results/study2/study2_results.json"), R("results/study3/study3_results.json"),
                     R("results/study3/planE_results.json"), R("results/study2/study2_fhrma.json"))
    sec = {}
    sec["prereg"] = ("Pre-specification record", [
        "- Plan 1 (end-to-end model, `docs/analiz_plani_v2.md`): git tag `analiz-plani-v2.0`, frozen before any NeSy-E2E model was trained.",
        "- Plan 2 and Addendum 2: SHA-256 checksums logged in `experiments/log.csv` before the runs they cover.", ""]
        + plan_checksums("paper1") + [""] + deviations("paper1"))

    sec["figo"] = figo_section()
    FW = R("results/study3/foldwise_events.json")
    srcs = (("NeSy-E2E (end-to-end)", "nesy2", S2["events"], "nesy2"), ("Sequential detector", "det", S3["detector_events"], "det"),
            ("Rule-guided detector", "detk", E["detk_events"], "det"), ("Rule-based extractor", "rule", S3["detector_events"], "rule"))
    b = ["Main analysis: each outer-test record thresholded with its own fold's validation-selected threshold; F1 pooled over records "
         "within each repeat and averaged over the 5 repeats (Se and PPV likewise). Pre-specified analysis: median of the 50 fold "
         "thresholds applied to repeat-averaged frame probabilities (indirectly dependent on the record's own labels; deviation 15 in S[prereg]). "
         "95% CIs: paired record bootstrap, 2000 resamples, with fixed out-of-fold predictions.", "",
         "| Source | Event | Se | PPV | F1, fold-specific threshold (95% CI) | F1, pre-specified median threshold (95% CI) | Expert events |",
         "|---|---|---|---|---|---|---|"]
    for name, fk, src, key in srcs:
        for ev in ("ACC", "DEC", "UC"):
            v, w = FW["events"][f"{fk}|{ev}"], src[f"{key}|{ev}"]
            b.append(f"| {name} | {ev} | {f3(v['se'])} | {f3(v['ppv'])} | {ci(v['f1_ci'])} | {ci(w['f1_ci'])} | {w['n_ref']} |")
    for ev in ("ACC", "DEC", "UC"):
        w = S2["events"][f"nomask|{ev}"]
        b.append(f"| NeSy-E2E without quality mask | {ev} | {f3(w['se'])} | {f3(w['ppv'])} | — | {ci(w['f1_ci'])} | {w['n_ref']} |")
    b += ["", "For NeSy-E2E without quality mask, Se and PPV are from the pre-specified median threshold; fold-specific outputs were not "
          "computed for this ablation."]
    t, sc = FW["tests"], FW["secondary_record_median_threshold"]
    b += ["", "**Pre-specified comparisons under both protocols** (difference in F1, learned source minus extractor; Holm adjustment within the pre-specified families):", "",
          "| Test | Fold-specific: difference (95% CI), Holm p | Pre-specified median threshold: difference (95% CI), Holm p | Secondary, record-specific median of its own fold thresholds: difference (95% CI), unadjusted p |",
          "|---|---|---|---|"]
    pre = {"H1a_DEC": S2["H1"]["H1a_DEC"], "H1b_ACC": S2["H1"]["H1b_ACC"], "B1a_DEC": S3["superiority"]["B1a_DEC"],
           "B1b_ACC": S3["superiority"]["B1b_ACC"], "E1a_DEC": E["E1"]["E1a_DEC"], "E1b_ACC": E["E1"]["E1b_ACC"]}
    secm = {"H1a_DEC": "H1_DEC", "H1b_ACC": "H1_ACC", "B1a_DEC": "B1_DEC", "B1b_ACC": "B1_ACC", "E1a_DEC": "E1_DEC", "E1b_ACC": "E1_ACC"}
    lab = {"H1a_DEC": "H1a, NeSy-E2E vs extractor, DEC", "H1b_ACC": "H1b, NeSy-E2E vs extractor, ACC", "B1a_DEC": "B1a, sequential detector vs extractor, DEC",
           "B1b_ACC": "B1b, sequential detector vs extractor, ACC", "E1a_DEC": "E1a, rule-guided detector vs extractor, DEC", "E1b_ACC": "E1b, rule-guided detector vs extractor, ACC"}
    for k in ("H1a_DEC", "H1b_ACC", "B1a_DEC", "B1b_ACC", "E1a_DEC", "E1b_ACC"):
        a, p_, c = t[k], pre[k], sc[secm[k]]
        b.append(f"| {lab[k]} | {a['diff']:+.3f} ({a['lo']:+.3f} to {a['hi']:+.3f}), {fp(a['p_holm'], '0.003')} | "
                 f"{p_['diff']:+.3f} ({p_['lo']:+.3f} to {p_['hi']:+.3f}), {fp(p_['p_holm'], '0.003')} | {c['diff']:+.3f} ({c['lo']:+.3f} to {c['hi']:+.3f}), {fp(c['p'])} |")
    b += ["", "The secondary record-specific analysis was added after the fold-specific results were seen and is descriptive (deviation 18 in S[prereg])."]
    sec["eventfull"] = ("Event-level fidelity, full table (CTU-UHB, outer-test records, quality region)", b)

    b = ["| Source | ACC F1 | DEC F1 | UC F1 |", "|---|---|---|---|"]
    for name, fk, _, _ in srcs:
        b.append(f"| {name} | " + " | ".join(f3(FW['events'][f'{fk}_iou50|{e}']['f1']) for e in ("ACC", "DEC", "UC")) + " |")
    b += ["", "Fold-specific thresholds, IoU ≥ 0.5. Rule-guided detector minus extractor: "
          + "; ".join(f"{e} {t[f'detk_minus_rule_iou50|{e}']['diff']:+.3f} ({t[f'detk_minus_rule_iou50|{e}']['lo']:+.3f} to {t[f'detk_minus_rule_iou50|{e}']['hi']:+.3f})" for e in ("ACC", "DEC", "UC")) + "."]
    sec["iou"] = ("Stricter matching (IoU ≥ 0.5)", b)

    b = ["| Source | Event | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 |", "|---|---|---|---|---|---|---|"]
    for name, fk, _, _ in srcs[:3]:
        for ev in ("ACC", "DEC"):
            b.append(f"| {name} | {ev} | " + " | ".join(f3(x) for x in FW["events"][f"{fk}|{ev}"]["f1_per_repeat"]) + " |")
    sec["perrepeat"] = ("Per-repeat event F1 (fold-specific thresholds)", b)

    b = ["| Concept | Positives | NeSy-E2E AUROC (κ) | Weak-label NeSy AUROC (κ) | Weak-label NeSy without concept supervision AUROC | Rule-based weak label κ |",
         "|---|---|---|---|---|---|"]
    for k, v in S2["window_concepts"].items():
        name = f"{cap(PRETTY.get(k, k))} (`{k}`)"
        if v.get("reported"):
            b.append(f"| {name} | {v['n_pos']} | {f3(v['auroc_nesy2'])} ({v['kappa_nesy2']:.2f}) | {f3(v['auroc_nesy1'])} ({v['kappa_nesy1']:.2f}) | "
                     f"{f3(v['auroc_nesy1_nocsup'])} | {v['kappa_weak']:.2f} |")
        else:
            b.append(f"| {name} | {v['n_pos']} | not evaluated (< 20 positives) | | | |")
    b += ["", "Weak-label NeSy without concept supervision: the benchmark ablation with the weak-label concept loss switched off (weight 0)."]
    sec["window"] = ("Window concepts against expert-derived labels (n = 340 annotated, quality-eligible records)", b)

    b = []
    for kind in ("nesy2", "rule"):
        d = S2["dec_type"][kind]
        conf = d["confusion"]                      # pandas crosstab(expert, predicted).to_dict(): {predicted: {expert: n}}
        preds = sorted(conf)
        hdr = ["U" if x == "other" else x for x in preds]
        experts = sorted({e for col in conf.values() for e in col})
        b += [f"**{'NeSy-E2E' if kind == 'nesy2' else 'Rule-based extractor'}** — matched pairs {d['n_pairs']}, macro-F1 {f3(d['macro_f1'])}; "
              "recall " + ", ".join(f"{t} {d['per_type_recall'][t]:.2f}" for t in d["per_type_recall"]), "",
              "| Expert type (rows) / predicted type (columns) | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(preds)]
        for e in experts:
            b.append(f"| {e} | " + " | ".join(str(conf[p].get(e, 0)) for p in preds) + " |")
        b.append("")
    b.append("E, early; L, late; P, prolonged; V, variable; U, unclassified deceleration (see S[figo]).")
    sec["dectype"] = ("Deceleration-type confusion among matched decelerations (exploratory; pre-specified median threshold)", b)

    b = ["| Rule | Median | 2.5% | 97.5% |", "|---|---|---|---|"]
    for k in S2["H2"]["rule_weights_median"]:
        b.append(f"| {cap(RULE_TXT.get(k, k))} (`{k}`) | {f3(S2['H2']['rule_weights_median'][k])} | {f3(S2['H2']['rule_weights_p2_5'][k])} | {f3(S2['H2']['rule_weights_p97_5'][k])} |")
    sec["ruleweights"] = ("Rule weights of the end-to-end model (medians over 50 outer folds; 2.5–97.5th percentiles)", b)

    b = ["| Model | AUROC rebuilt, no intervention | AUROC intervened | Log-loss rebuilt | Log-loss intervened | Mean \\|Δ risk\\| |", "|---|---|---|---|---|---|"]
    for m, nm in (("nesy2_W60", "NeSy-E2E"), ("nesy_ens_W60", "Weak-label NeSy")):
        v = S2["leakage_intervention"][m]
        b.append(f"| {nm} | {f3(v['auroc_model'])} | {f3(v['auroc_intervened'])} | {f3(v['logloss_model'])} | {f3(v['logloss_intervened'])} | {f3(v['mean_abs_dp'])} |")
    ic = R("results/study2/intervention_check.json")
    b += ["", "Fidelity of the rebuild (deviation 17): without intervention, the rebuilt risks agreed with the recorded model risks with "
          + "; ".join(f"{nm} Spearman {ic[m]['spearman']:.2f}, mean absolute difference {ic[m]['mean_abs_diff']:.3f}, AUROC "
                      f"{ic[m]['auroc_rebuilt']:.3f} rebuilt vs {ic[m]['auroc_recorded']:.3f} recorded"
                      for m, nm in (("nesy2_W60", "NeSy-E2E"), ("nesy_ens_W60", "Weak-label NeSy")))
          + ". The rebuild error is of the same size as the approximate intervention effect, so the approximate analysis can exclude only a large dependence."]
    i = R("results/study2/posthoc_member_intervention.json")
    b = ["**Original approximate intervention (fold models rebuilt from member-averaged logits and rule weights; pre-specified)**", "", *b, "",
         "**Member-level intervention (post hoc)**", "",
         f"We replayed {i['members']} saved NeSy-E2E members. The mean absolute discrepancy from the stored member predictions before substitution was "
         f"{i['baseline_member_mean_abs_error']:.2e}. Baseline pooled AUROC was {i['baseline_auc']:.3f}. Expert binary values were inserted as logits ±9.2. "
         "Member-specific calibration was applied before averaging. The 95% AUROC difference intervals use 2,000 paired outcome-stratified record resamples.", "",
         "| Substitution | AUROC after | ΔAUROC (95% CI) | Mean / median absolute change (pp) | Absolute change ≥ 5 pp |",
         "|---|---|---|---|---|"]
    for key, nm in (("all", "All eight expert-referenced concepts"), ("acc", "Acceleration only"), ("dec", "Deceleration family")):
        v = i["changes"][key]
        lo, hi = v["ci95_auc"]
        b.append(f"| {nm} | {v['auc']:.3f} | {v['delta_auc']:+.3f} ({lo:+.3f} to {hi:+.3f}) | "
                 f"{100*v['mean_abs_risk']:.2f} / {100*v['percentiles_abs_risk'][0]:.2f} | {v['n_abs_risk_ge_5pp']}/{i['n']} |")
    b += ["", "pp, percentage points. The five-point cutoff describes the distribution. Results for the weak-label comparator are available for the approximate intervention."]
    sec["intervention"] = ("Concept intervention test (340 recordings, 16 events)", b)

    c = R("results/study2/posthoc_controlled_loss.json")
    cfg = c["configuration"]
    b = [f"One ten-fold repeat, one member per arm per fold and {c['n_records']} test recordings. "
         f"Both arms used width {cfg['width']}, {cfg['n_stages']} stages, kernel {cfg['kernel']}, dropout {cfg['drop']}, "
         f"AdamW, learning rate {cfg['lr']}, weight decay {cfg['wd']}, batch size {cfg['bs']} and {c['epochs']} fixed epochs. "
         "Inputs, supervised concept losses, seed, minibatch ordering and fitting records were shared. "
         "The arms differed only by the weighting of the acidaemia loss (0 or 1). Thresholds were selected separately on the same held-out validation records. "
         "Intervals use 2,000 paired record bootstraps with fitted models held fixed.", "",
         "| Event match | Event | F1 without outcome loss | F1 with outcome loss | Difference (95% CI) |",
         "|---|---|---|---|---|"]
    for key, nm in (("any_overlap", "Any overlap"), ("iou_50", "IoU ≥ 0.5")):
        for event, v in c["outcomes"][key].items():
            lo, hi = v["ci95"]
            b.append(f"| {nm} | {event} | {v['no_outcome_f1']:.3f} | {v['with_outcome_f1']:.3f} | "
                     f"{v['difference']:+.3f} ({lo:+.3f} to {hi:+.3f}) |")
    sec["controlled"] = ("Controlled ablation of acidaemia outcome loss", b)

    b = ["| Source | ACC Se | ACC PPV | ACC F1 | DEC Se | DEC PPV | DEC F1 |", "|---|---|---|---|---|---|---|"]
    rows = [("NeSy-E2E", F2["ACC"]["nesy2"], F2["DEC"]["nesy2"]), ("Sequential detector", S3["detector_fhrma"]["ACC"]["det"], S3["detector_fhrma"]["DEC"]["det"]),
            ("Rule-guided detector", E["E2_fhrma"]["ACC"]["det"], E["E2_fhrma"]["DEC"]["det"]), ("Rule-based extractor", E["E2_fhrma"]["ACC"]["rule"], E["E2_fhrma"]["DEC"]["rule"])]
    for nm, a, d in rows:
        b.append(f"| {nm} | {f3(a['se'])} | {f3(a['ppv'])} | {f3(a['f1'])} | {f3(d['se'])} | {f3(d['ppv'])} | {f3(d['f1'])} |")
    sec["fhrma"] = ("External expert consensus (FHRMA, 66 public recordings)", b)

    k3 = S2["K3_reference_audit"]
    LBL = {"TC": "tachycardia", "BC": "bradycardia"}
    kd = R("results/study2/k3_and_diag_numbers.json")
    b = [f"Spearman correlation of expert ACC+DEC density with FHR loss in the 60-min window: {k3['spearman_density_vs_missing']:.2f}.", "",
         "| FHR-loss tertile | n | Acidaemic | Median density, pH < 7.05 (events/h) | Median density, pH ≥ 7.05 (events/h) | Mann–Whitney p |",
         "|---|---|---|---|---|---|"]
    for t, v in k3["by_tertile"].items():
        b.append(f"| {t} | {v['n']} | {v['n_pos']} | {v['med_density_pos']} | {v['med_density_neg']} | {v['mannwhitney_p']:.2f} |")
    b += ["", f"Recordings failing the record-level quality criterion: {kd['fail_record_quality_acid']}/{kd['annotated_acid']} acidaemic vs "
          f"{kd['fail_record_quality_nonacid']}/{kd['annotated_nonacid']} non-acidaemic annotated recordings. Mean share of the window excluded "
          f"from expert supervision: {k3['masked_frac_mean_by_outcome']['1']:.2f} vs {k3['masked_frac_mean_by_outcome']['0']:.2f}.", "",
          f"Annotation inventory: {kd['romagnoli_total_annotated_episodes']:,} annotated episodes "
          f"({', '.join(LBL.get(k, k) + f' {v:,}' for k, v in kd['romagnoli_by_label'].items())}).", "",
          f"Exploratory diagnostic check of the window acceleration concept (deviation 8 in S[prereg]): per-fold AUROC "
          f"{kd['h1c_perfold_auroc_mean']:.3f} (SD {kd['h1c_perfold_auroc_sd']:.3f}) over {kd['h1c_n_folds']} folds."]
    sec["refaudit"] = ("Reference-standard audit", b)

    sec["hyper"] = ("Hyper-parameter search spaces", [
        "- CNN encoder (all learned models): width {16, 32, 48}; stages {3, 4}; kernel {5, 7, 9}; dropout {0.1, 0.3, 0.5}; learning rate {1e-3, 3e-4}; weight decay {1e-4, 1e-3, 1e-2}; batch 32; 6 random configurations; 3 inner folds; early stopping (≤ 60 epochs, patience 10).",
        "- NeSy-E2E selection: inner outcome AUROC. Detectors: inner mean ACC/DEC event F1 (best grid threshold).",
        "- Event thresholds: grid 0.10–0.90 (step 0.05), chosen on the 15% validation split by event F1; post-processing merges gaps ≤ 1 bin, minimum length 4 bins (ACC/DEC) or 8 bins (UC).",
        "- Transparent risk models on detector concepts: L2 logistic regression with cubic B-splines, C log-uniform on [10^−3^, 10^1^], knots {3, 4, 5}, 30 configurations."])
    if sections_only:
        return sec
    return assemble("paper1", "Concept-fidelity sections (end-to-end model, detectors and rule-based extractor).", sec)


def paper2(sections_only=False):
    S3, D, X, T, E4 = (R("results/study3/study3_results.json"), R("results/study3/planD_results.json"), R("results/study3/extra_results.json"),
                       R("results/study3/tripod_results.json"), R("results/study3/transport_ek4_results.json"))
    M = json.loads((ROOT / "results/study3/transp_lr_final_model.json").read_text())
    sec = {}
    sec["prereg"] = ("Pre-specification record", ["SHA-256 checksums of the plan and its addenda were logged in `experiments/log.csv` "
                                                 "before the runs they cover.", ""] + plan_checksums("paper2") + [""] + deviations("paper2"))

    sec["figo"] = figo_section()
    defs = {"baseline": "Median of the robust moving baseline (bpm)", "stv_ms": "Short-term variability, Dawes–Redman style, 3.75-s epochs (ms)",
            "ltv_ms": "Long-term variability (ms)", "var_amp_median": "Median per-minute bandwidth (P95–P5) after removing events (bpm)",
            "reduced_var_min": "Minutes with bandwidth < 5 bpm", "increased_var_min": "Minutes with bandwidth > 25 bpm",
            "n_acc": "Accelerations (> 15 bpm, > 15 s)", "n_dec": "Decelerations (> 15 bpm, > 15 s)", "n_dec_early": "Early decelerations",
            "n_dec_late": "Late decelerations", "n_dec_variable": "Variable decelerations", "n_dec_prolonged": "Decelerations > 3 min",
            "n_dec_other": "Decelerations not classifiable by FIGO timing rules (including those without a valid contraction signal)",
            "dec_total_min": "Total deceleration time (min)",
            "dec_max_s": "Longest deceleration (s)", "dec_area": "Total deceleration area (bpm·s)", "dec_max_depth": "Deepest deceleration (bpm)",
            "late_frac_contr": "Fraction of contractions with a late deceleration", "variable_frac_contr": "Fraction with a variable deceleration",
            "early_frac_contr": "Fraction with an early deceleration", "any_dec_frac_contr": "Fraction with any deceleration",
            "sinusoidal_min": "Minutes classified as sinusoidal", "uc_per_10min": "Contractions per 10 min",
            "vlf": "log FHR power, 0–0.03 Hz (very low frequency)", "lf": "log FHR power, 0.03–0.15 Hz (low frequency)",
            "sd2": "Poincaré SD2 of successive 1-Hz FHR values (long-term variability; bpm)",
            "fhr_missing": "Fraction of missing FHR samples in the window", "uc_missing": "Fraction of missing UC samples",
            "fhr_longest_gap_s": "Longest FHR gap (s)", "uc_longest_gap_s": "Longest UC gap (s)"}
    uc7 = {"uc_per_10min", "late_frac_contr", "variable_frac_contr", "early_frac_contr", "any_dec_frac_contr", "uc_missing", "uc_longest_gap_s"}
    b = ["| Variable | Definition |", "|---|---|"]
    for c in M["coefficients"]:
        b.append(f"| `{c['variable']}`{' †' if c['variable'] in uc7 else ''} | {defs.get(c['variable'], '')} |")
    b += ["", "† One of the 7 contraction variables removed in the leaner models (contraction rate, deceleration–contraction "
          "association and contraction-signal quality). Deceleration typing (early, late, variable, unclassified) still uses the contraction signal.", "",
          "W60 uses the last 60 min of the available trace. All 552 public headers give `Sig2Birth = 0`. "
          "The database report allows recordings to end up to 30 min before birth, so exact alignment could not be established. "
          "The report describes 16 selected caesareans with pH ≤ 7.05 among 46 caesareans, whereas 15 of the 46 public headers meet that threshold. "
          "The study analysed 536 recordings, with 38 events at its primary cutoff of pH < 7.05."]
    sec["vardefs"] = ("Variable definitions, analysis window and cohort", b)

    b = [f"L2 logistic regression on standardised variables, C = {M['C']:.4f} (median of the 50 fold-selected values); "
         f"intercept (class-balanced fit) {M['intercept_balanced']:.4f}; {M['prior_correction']}. Missing values: median imputation with the values below.", "",
         "| Variable | Imputation value | Mean | SD | Coefficient per SD (95% bootstrap CI) |", "|---|---|---|---|---|"]
    for c in M["coefficients"]:
        mark = " ‡" if c["variable"] == "sinusoidal_min" else ""
        b.append(f"| `{c['variable']}`{mark} | {c['median_imputation']:.4g} | {c['mean']:.4g} | {c['sd']:.4g} | {c['coef_per_sd']:+.3f} ({c['coef_ci95'][0]:+.3f} to {c['coef_ci95'][1]:+.3f}) |")
    b += ["", "‡ Zero in all 536 recordings; the SD of 1 is the scaler's placeholder for a constant variable, so the term does not contribute."]
    sec["finalmodel"] = ("Full specification of the refitted linear model", b)

    NM = {"cnn_bayes": "CNN ensemble", "figo_lr": "FIGO-feature model", "georgoulas": "HRV support-vector model",
          "transp_lr": "Linear model", "transp_gam": "Additive model", "transp_lr_nouc": "Linear, no UC variables",
          "transp_gam_nouc": "Additive, no UC variables", "cnn": "CNN ensemble"}
    SET = {"W60": "Primary (last 60 min)", "W30": "Last 30 min", "W90": "Last 90 min", "S1_60": "First stage, last 60 min",
           "all552": "All 552 recordings", "ph715": "pH ≤ 7.15", "bdecf12": "BDecf ≥ 12 mmol/L", "apgar5": "5-min Apgar < 7"}
    b = ["| Setting | n / events | CNN ensemble | FIGO-feature model | Linear model | Additive model | Single-run linear | Single-run CNN |",
         "|---|---|---|---|---|---|---|---|"]
    for s in ("W60", "W30", "W90", "S1_60", "all552", "ph715", "bdecf12", "apgar5"):
        r = D[s]
        b.append(f"| {SET[s]} | {r['n']} / {r['n_pos']} | {ci(r['cnn']['auroc'])} | {ci(r['figo']['auroc'])} | {ci(r['transp_lr']['auroc'])} | "
                 f"{ci(r['transp_gam']['auroc'])} | {r['transp_lr']['per_repeat'][0]:.3f} ± {r['transp_lr']['per_repeat'][1]:.3f} | "
                 f"{r['cnn']['per_repeat'][0]:.3f} ± {r['cnn']['per_repeat'][1]:.3f} |")
    b += ["", "In the seven additional settings the transparent and FIGO-feature models were fully re-tuned; as pre-specified, the CNN "
          "ensemble used the configuration selected most often in the primary analysis."]
    sec["robust"] = ("Robustness settings (AUROC, 95% CI; single-run AUROC mean ± sample SD over 5 repeats)", b)

    five = ["cnn_bayes", "figo_lr", "georgoulas", "transp_lr", "transp_gam"]
    b = ["| Model | Cal. slope | Cal. intercept | Sens. at 85% spec. | Sens. at 90% spec. |", "|---|---|---|---|---|"]
    for m in five:
        v = X[m]
        b.append(f"| {NM[m]} | {v['cal_slope']:.2f} | {v['cal_intercept']:.2f} | {v['sens_at_spec85']:.2f} | {v['sens_at_spec90']:.2f} |")
    b += ["", "On each outer-training validation subset, the CNN, additive and logistic models used temperature scaling "
          "after class-prior correction. The HRV support-vector model used Platt scaling of the uncorrected decision scores. "
          "The calibration and operating-point summaries in the table were added after the plan was frozen (deviation 11 in S[prereg]).", "",
          "**Continuous pH.** Harrell's C used the repeat-averaged out-of-fold risks. It gives credit for assigning higher risk to "
          "the lower-pH member of each pair with different pH, with half credit for a tied risk. The estimates were "
          + ", ".join(f"{S3['discrimination'][m]['harrell_c']:.3f} ({NM[m]})" for m in five) + "."]
    sec["calib"] = ("Calibration, operating points and continuous pH (primary analysis)", b)

    b = ["| Threshold | Treat all | " + " | ".join(NM[m] for m in five) + " |", "|---|---|" + "---|" * len(five)]
    for t in ("0.05", "0.1", "0.15", "0.2", "0.25", "0.3"):
        b.append(f"| {t} | {X['treat_all_net_benefit'][t]:.4f} | " + " | ".join(f"{X[m]['net_benefit'][t]:.4f}" for m in five) + " |")
    sec["netbenefit"] = ("Net benefit at selected thresholds (primary analysis)", b)

    a = R("results/study3/posthoc_transparent.json")
    b = ["For each fold, the decision threshold was the empirical 90th percentile of negative predictions in its validation split. "
         "The values below pool test decisions over the five repeats. They describe the performance of the threshold-selection procedure.", "",
         "| Model | Test sensitivity | Test specificity |", "|---|---|---|"]
    names = {"cnn": "CNN ensemble", "gam": "Additive model", "lr": "Linear model",
             "hrv_lr": "Three-variable HRV linear model", "figo_lr": "FIGO-feature model"}
    for k, v in a["validation_threshold_test_points"].items():
        b.append(f"| {names[k]} | {v['sensitivity']:.3f} | {v['specificity']:.3f} |")
    b += ["", f"FHR loss alone had AUROC {a['missing_only_auc']:.3f} as a descriptive ranking score.", "",
          "| Hypothetical threshold | Comparison | Net benefit difference (95% CI) |", "|---|---|---|"]
    for threshold, comps in a["linear_minus_comparator_net_benefit"].items():
        for k, v in comps.items():
            lo, hi = v["ci95"]
            b.append(f"| {float(threshold):.2f} | Linear minus {names[k]} | {v['diff']:+.4f} ({lo:+.4f} to {hi:+.4f}) |")
    b += ["", "Intervals are from 2,000 paired record bootstrap draws on fixed out-of-fold probabilities. "
          "The estimates use the CTU-UHB sample event rate and are descriptive of this selected cohort."]
    sec["review"] = ("Validation-selected thresholds and paired net benefit", b)

    b = ["Shift A: train on vaginal births (498 recordings, 25 events), test on caesareans (38, 13 events). Shift B: train on FHR loss ≤ median "
         "(268, 9 events), test on the rest (268, 29 events). "
         "UC, uterine contractions. Some features still require UC timing for deceleration typing.", "",
         "| Shift | Model | AUROC | Intercept raw | Brier raw | Brier, intercept update | Slope after intercept update | Brier, intercept + slope |", "|---|---|---|---|---|---|---|---|"]
    TRN = R("results/study3/transport_results.json")
    for s, lab in (("C1", "A"), ("C3", "B")):
        for m, v in E4[s].items():
            if isinstance(v, dict):
                raw = TRN.get(s, {}).get(m, {})
                itc = f"{raw['cal_intercept']:+.2f}" if isinstance(raw, dict) and "cal_intercept" in raw else "—"
                b.append(f"| {lab} | {NM.get(m, m)} | {v['auroc'][0]:.3f} | {itc} | {v['brier_raw']:.4f} | {v['recal_intercept']['brier']:.4f} | "
                         f"{v['recal_intercept']['cal_slope']:.2f} | {v['recal_intercept_slope']['brier']:.4f} |")
    sec["shift"] = ("Distribution shifts: recalibration within the target group (leave-one-out)", b)

    im = S3["informative_missingness"]
    sec["ablation"] = ("Variable-group ablations and complementarity (primary analysis)", [
        f"- Full additive model minus additive model without HRV: {im['transp_minus_nohrv'][0]:+.3f} "
        f"({im['transp_minus_nohrv'][1]:+.3f} to {im['transp_minus_nohrv'][2]:+.3f}).",
        f"- Full additive model minus additive model without signal quality: {im['transp_minus_noq'][0]:+.3f} "
        f"({im['transp_minus_noq'][1]:+.3f} to {im['transp_minus_noq'][2]:+.3f}).",
        f"- Linear model without the 7 contraction variables: AUROC {ci(E4['cv_transp_lr_nouc']['auroc'])}; additive model without them: {ci(E4['cv_transp_gam_nouc']['auroc'])}.",
        f"- Average of additive model and CNN ensemble (pre-specified, descriptive): AUROC {S3['complementarity_avg_transp_cnn']['auroc'][0]:.3f}; "
        f"average of linear model and CNN ensemble (added after the results, deviation 18 in S[prereg]): {X['avg_transp_lr_cnn']['auroc']:.3f}."])

    sg = T["subgroups"]
    SG = {"nulliparous": "Nulliparous", "parous": "Parous", "maternal_age_lt35": "Maternal age < 35 y", "maternal_age_ge35": "Maternal age ≥ 35 y",
          "gest_lt41": "Gestation < 41 wk", "gest_ge41": "Gestation ≥ 41 wk", "induced": "Induced labour", "spontaneous": "Spontaneous labour",
          "vaginal": "Vaginal delivery", "caesarean": "Caesarean delivery"}
    b = ["| Subgroup | n / events | CNN ensemble | Linear model | Additive model |", "|---|---|---|---|---|"]
    for k, v in sg.items():
        if all(isinstance(v[m], list) for m in ("cnn_bayes", "transp_lr", "transp_gam")):
            b.append(f"| {SG.get(k, k)} | {v['n']} / {v['events']} | {ci(v['cnn_bayes'])} | {ci(v['transp_lr'])} | {ci(v['transp_gam'])} |")
    b += ["", "Descriptive TRIPOD+AI addition (deviation 12 in S[prereg])."]
    sec["subgroup"] = ("Subgroup discrimination (AUROC, 95% CI; descriptive)", b)

    decl = "Acknowledgements and declarations"
    rows = [("1", "Title", "Title"), ("2", "Abstract", "Abstract"), ("3a", "Context, rationale, existing models", "§1"),
            ("3b", "Population, intended purpose and users", "§1; §4.2"), ("3c", "Health inequalities", "§4.4"), ("4", "Objectives", "§1 (Aim)"),
            ("5a", "Data sources, representativeness", "§2.1"), ("5b", "Dates", "§2.1 (2010–2012)"), ("6a", "Setting, centres", "§2.1 (single centre)"),
            ("6b", "Eligibility", "§2.1"), ("6c", "Treatments", "§4.4 (not modelled)"), ("7", "Pre-processing, quality checks", "§2.1"),
            ("8a", "Outcome definition and timing", "§2.1"), ("8b", "Outcome assessors", "Not applicable (laboratory measurement)"),
            ("8c", "Blinding of outcome", "Not applicable (objective measurement)"), ("9a", "Choice of predictors", "§2.2"),
            ("9b", "Predictor definitions", "§2.2; S[vardefs]"), ("9c", "Predictor assessors", "Not applicable (deterministic code)"),
            ("10", "Study size", "§2.1 (all eligible recordings of the database)"), ("11", "Missing data", "§2.1; §2.3"),
            ("12a", "Data use and partitioning", "§2.4"), ("12b", "Predictor handling", "§2.3"), ("12c", "Model type, tuning, internal validation", "§2.3–2.4"),
            ("12d", "Heterogeneity across clusters", "Not applicable (single centre)"), ("12e", "Performance measures", "§2.5"),
            ("12f", "Model updating", "§2.5; §3.7 (within-target recalibration, descriptive)"), ("12g", "How predictions were computed", "S[finalmodel]; code repository"),
            ("13", "Class imbalance", "§2.4"), ("14", "Fairness", "§4.4"), ("15", "Model output, thresholds", "§2.4 (calibrated probability); §3.2 and S[review] (validation-selected operating points, descriptive; no clinical threshold proposed)"),
            ("16", "Development vs evaluation data", "Internal validation only (§2.4)"), ("17", "Ethics", decl),
            ("18a–f", "Funding, conflicts, protocol, registration, data, code", f"{decl}; Data and code availability; S[prereg]"),
            ("19", "Patient and public involvement", decl), ("20a", "Participant flow", "§2.1; §3.1"), ("20b", "Characteristics", "Table 3"),
            ("20c", "Comparison with development data", "Not applicable"), ("21", "Numbers per analysis", "§2.4"), ("22", "Full model specification", "S[finalmodel]"),
            ("23a", "Performance with CIs, subgroups", "Tables 4–6; S[calib]; S[robust]; S[subgroup]"), ("23b", "Heterogeneity across clusters", "Not applicable"),
            ("24", "Model updating results", "§3.7; S[shift]"), ("25", "Interpretation", "§4.1–4.3"), ("26", "Limitations", "§4.4"),
            ("27a", "Handling poor-quality inputs", "§2.1 (exclusion rule); §3.7"), ("27b", "User interaction", "§4.2"),
            ("27c", "Next steps", "§4.4–4.5")]
    sec["tripod"] = ("TRIPOD+AI checklist (version 11 January 2024)", ["| Item | Topic | Where reported |", "|---|---|---|"]
                     + [f"| {a} | {b_} | {c} |" for a, b_, c in rows])
    if sections_only:
        return sec
    return assemble("paper2", "Outcome-model sections (transparent models, FIGO-feature model and CNN ensemble).", sec)


def combined():
    """Supplement of this paper (section numbering from manuscript/combined/combined_paper.md)."""
    sec = {**paper2(True), **paper1(True)}
    sec["prereg"] = ("Pre-specification record", [
        "- Plan 1 (end-to-end model, `docs/analiz_plani_v2.md`): git tag `analiz-plani-v2.0`, frozen before any NeSy-E2E model was trained.",
        "- Plan 2 and Addenda 1–4: SHA-256 checksums logged in `experiments/log.csv` before the runs they cover.", ""]
        + plan_checksums("combined") + [""] + deviations("combined"))
    decl = "Ethics statement; Funding; Declaration of competing interest"
    rows = [("1", "Title", "Title"), ("2", "Abstract", "Abstract"), ("3a", "Context, rationale, existing models", "§1.1–1.3"),
            ("3b", "Population, intended purpose and users", "§1.4; §4.5 (retrospective research benchmark)"), ("3c", "Health inequalities", "§4.5"),
            ("4", "Objectives", "§1.4"), ("5a", "Data sources, representativeness", "§2.1"), ("5b", "Dates", "§2.1 (2010–2012)"),
            ("6a", "Setting, centres", "§2.1 (single centre)"), ("6b", "Eligibility", "§2.1"), ("6c", "Treatments", "§4.5 (not modelled)"),
            ("7", "Pre-processing, quality checks", "§2.1–2.2"), ("8a", "Outcome definition and timing", "§2.1"),
            ("8b", "Outcome assessors", "Not applicable (laboratory measurement)"), ("8c", "Blinding of outcome", "Not applicable (objective measurement)"),
            ("9a", "Choice of predictors", "§2.3"), ("9b", "Predictor definitions", "§2.3; S[vardefs]; S[figo]"),
            ("9c", "Predictor assessors", "Not applicable (deterministic code)"), ("10", "Study size", "§2.1 (all eligible recordings of the database)"),
            ("11", "Missing data", "§2.1; Table 1 (median imputation)"), ("12a", "Data use and partitioning", "§2.4"), ("12b", "Predictor handling", "§2.3 (Table 1)"),
            ("12c", "Model type, tuning, internal validation", "§2.3–2.4; S[hyper]"), ("12d", "Heterogeneity across clusters", "Not applicable (single centre)"),
            ("12e", "Performance measures", "§2.5"), ("12f", "Model updating", "§2.5; S[shift] (within-target recalibration, descriptive)"),
            ("12g", "How predictions were computed", "S[finalmodel]; code repository"), ("13", "Class imbalance", "§2.3–2.4 (balanced weights, prior correction)"),
            ("14", "Fairness", "§4.5"), ("15", "Model output, thresholds", "§2.4 (calibrated probability); §3.6 and S[review] (no clinical threshold proposed)"),
            ("16", "Development vs evaluation data", "Internal validation only (§2.4)"), ("17", "Ethics", "Ethics statement"),
            ("18a–f", "Funding, conflicts, protocol, registration, data, code", f"{decl}; Protocol and registration; Data and code availability; S[prereg]"),
            ("19", "Patient and public involvement", "Ethics statement"), ("20a", "Participant flow", "§2.1; §3.1"), ("20b", "Characteristics", "Table 3"),
            ("20c", "Comparison with development data", "Not applicable"), ("21", "Numbers per analysis", "§2.4"), ("22", "Full model specification", "S[finalmodel]"),
            ("23a", "Performance with CIs, subgroups", "Tables 4–5; S[calib]; S[robust]; S[subgroup]"), ("23b", "Heterogeneity across clusters", "Not applicable"),
            ("24", "Model updating results", "§3.7; S[shift]"), ("25", "Interpretation", "§4.1–4.3"), ("26", "Limitations", "§4.5"),
            ("27a", "Handling poor-quality inputs", "§2.1 (exclusion rule); §2.2; §3.7"), ("27b", "User interaction", "§4.4"), ("27c", "Next steps", "§4.4–4.5")]
    sec["tripod"] = ("TRIPOD+AI checklist for the outcome models (version 11 January 2024)", ["| Item | Topic | Where reported |", "|---|---|---|"]
                     + [f"| {a} | {b_} | {c} |" for a, b_, c in rows])
    return assemble("combined", "Expert-referenced interpretability in intrapartum cardiotocography models for fetal acidaemia.", sec)


def main():
    (ROOT / "manuscript/supplement_paper1.md").write_text(paper1(), encoding="utf-8")
    (ROOT / "manuscript/supplement_paper2.md").write_text(paper2(), encoding="utf-8")
    (ROOT / "manuscript/combined/supplement_combined.md").write_text(combined(), encoding="utf-8")
    print("combined", numbers("combined"))
    print("paper1", numbers("paper1"))
    print("paper2", numbers("paper2"))


if __name__ == "__main__":
    main()
