# Supplementary material

*Expert-referenced interpretability in intrapartum cardiotocography models for fetal acidaemia.*

## S1. TRIPOD+AI checklist for the outcome models (version 11 January 2024)

| Item | Topic | Where reported |
|---|---|---|
| 1 | Title | Title |
| 2 | Abstract | Abstract |
| 3a | Context, rationale, existing models | §1.1–1.3 |
| 3b | Population, intended purpose and users | §1.4; §4.5 (retrospective research benchmark) |
| 3c | Health inequalities | §4.5 |
| 4 | Objectives | §1.4 |
| 5a | Data sources, representativeness | §2.1 |
| 5b | Dates | §2.1 (2010–2012) |
| 6a | Setting, centres | §2.1 (single centre) |
| 6b | Eligibility | §2.1 |
| 6c | Treatments | §4.5 (not modelled) |
| 7 | Pre-processing, quality checks | §2.1–2.2 |
| 8a | Outcome definition and timing | §2.1 |
| 8b | Outcome assessors | Not applicable (laboratory measurement) |
| 8c | Blinding of outcome | Not applicable (objective measurement) |
| 9a | Choice of predictors | §2.3 |
| 9b | Predictor definitions | §2.3; S3; S2 |
| 9c | Predictor assessors | Not applicable (deterministic code) |
| 10 | Study size | §2.1 (all eligible recordings of the database) |
| 11 | Missing data | §2.1; Table 1 (median imputation) |
| 12a | Data use and partitioning | §2.4 |
| 12b | Predictor handling | §2.3 (Table 1) |
| 12c | Model type, tuning, internal validation | §2.3–2.4; S4 |
| 12d | Heterogeneity across clusters | Not applicable (single centre) |
| 12e | Performance measures | §2.5 |
| 12f | Model updating | §2.5; S22 (within-target recalibration, descriptive) |
| 12g | How predictions were computed | S6; code repository |
| 13 | Class imbalance | §2.3–2.4 (balanced weights, prior correction) |
| 14 | Fairness | §4.5 |
| 15 | Model output, thresholds | §2.4 (calibrated probability); §3.6 and S19 (no clinical threshold proposed) |
| 16 | Development vs evaluation data | Internal validation only (§2.4) |
| 17 | Ethics | Ethics statement |
| 18a–f | Funding, conflicts, protocol, registration, data, code | Ethics statement; Funding; Declaration of competing interest; Protocol and registration; Data and code availability; S7 |
| 19 | Patient and public involvement | Ethics statement |
| 20a | Participant flow | §2.1; §3.1 |
| 20b | Characteristics | Table 3 |
| 20c | Comparison with development data | Not applicable |
| 21 | Numbers per analysis | §2.4 |
| 22 | Full model specification | S6 |
| 23a | Performance with CIs, subgroups | Tables 4–5; S9; S20; S21 |
| 23b | Heterogeneity across clusters | Not applicable |
| 24 | Model updating results | §3.7; S22 |
| 25 | Interpretation | §4.1–4.3 |
| 26 | Limitations | §4.5 |
| 27a | Handling poor-quality inputs | §2.1 (exclusion rule); §2.2; §3.7 |
| 27b | User interaction | §4.4 |
| 27c | Next steps | §4.4–4.5 |

## S2. Operationalisation of the FIGO 2015 criteria

The extractor implements numerical approximations to selected FIGO 2015 criteria. The table gives the clinical definition and its numerical implementation for each criterion.

| Criterion | FIGO 2015 | Operationalisation in this study | Difference |
|---|---|---|---|
| Baseline | Mean FHR level of the most horizontal, least oscillatory segments, estimated over 10-min periods | Iterative robust mean within ±5 min of each sample (values > 10 bpm from the running estimate excluded, two passes), evaluated every 10 s and interpolated | No visual selection of stable segments |
| Variability | Bandwidth amplitude of the FHR in 1-min segments; reduced < 5 bpm, increased > 25 bpm | Per-minute P95–P5 bandwidth after removing accelerations and decelerations; median over the window | P95–P5 replaces the visual peak-to-trough amplitude |
| Reduced variability, pathological | < 5 bpm for > 50 min (or > 3 min during decelerations) | Minutes with bandwidth < 5 bpm summed over the window (not necessarily continuous); > 50 min in the 60-min window | The criterion during decelerations is not implemented; in the 30- and 90-min sensitivity windows the duration is scaled with the window length |
| Increased (saltatory) variability | > 25 bpm for > 30 min | Minutes with bandwidth > 25 bpm summed; > 30 min in the 60-min window | As above |
| Acceleration | Abrupt increase > 15 bpm lasting > 15 s and < 10 min | Excursion > 15 bpm above the baseline for > 15 s and ≤ 10 min, extended to the baseline crossings | Abrupt onset is not checked |
| Deceleration | Decrease > 15 bpm lasting > 15 s | Excursion > 15 bpm below the baseline for > 15 s, extended to the baseline crossings; overlapping events are merged and measured once over the merged span | — |
| Early deceleration | Shallow, short, coincident with the contraction | Gradual (onset to nadir ≥ 30 s) with the nadir within ±20 s of a contraction peak | Depth and within-deceleration variability are not used |
| Late deceleration | Gradual onset and return, nadir > 20 s after the contraction peak | Gradual, with the nadir 20–60 s after the nearest contraction peak | — |
| Variable deceleration | Rapid drop (onset to nadir < 30 s) | Onset to nadir < 30 s | Within-deceleration variability is not used |
| Prolonged deceleration | > 3 min (pathological if > 5 min) | Duration > 3 min; > 5 min flagged as pathological | — |
| Repetitive decelerations | Associated with > 50% of contractions | Fraction of detected contractions with an associated deceleration (nadir within ±60 s of the peak) > 0.5 | Depends on contraction detection |
| Unclassified deceleration | Not a FIGO category | Gradual deceleration not associated with a contraction peak, or without a valid contraction signal | Introduced by the authors |
| Sinusoidal pattern | Smooth undulating pattern, 5–15 bpm, 3–5 cycles/min, > 30 min | 10-min segments with ≥ 50% of variability power at 3–5 cycles/min and amplitude 5–15 bpm; ≥ 30 min in the 60-min window | Spectral approximation |
| Contractions | Tachysystole: > 5 contractions in 10 min | Peaks of the 15-s smoothed UC signal (prominence ≥ 10 units, ≥ 60 s apart, half-width 15–180 s); contractions per 10 min | Tachysystole is not used as a rule |

## S3. Variable definitions, analysis window and cohort

| Variable | Definition |
|---|---|
| `baseline` | Median of the robust moving baseline (bpm) |
| `stv_ms` | Short-term variability, Dawes–Redman style, 3.75-s epochs (ms) |
| `ltv_ms` | Long-term variability (ms) |
| `var_amp_median` | Median per-minute bandwidth (P95–P5) after removing events (bpm) |
| `reduced_var_min` | Minutes with bandwidth < 5 bpm |
| `increased_var_min` | Minutes with bandwidth > 25 bpm |
| `n_acc` | Accelerations (> 15 bpm, > 15 s) |
| `n_dec` | Decelerations (> 15 bpm, > 15 s) |
| `n_dec_early` | Early decelerations |
| `n_dec_late` | Late decelerations |
| `n_dec_variable` | Variable decelerations |
| `n_dec_prolonged` | Decelerations > 3 min |
| `n_dec_other` | Decelerations not classifiable by FIGO timing rules (including those without a valid contraction signal) |
| `dec_total_min` | Total deceleration time (min) |
| `dec_max_s` | Longest deceleration (s) |
| `dec_area` | Total deceleration area (bpm·s) |
| `dec_max_depth` | Deepest deceleration (bpm) |
| `late_frac_contr` † | Fraction of contractions with a late deceleration |
| `variable_frac_contr` † | Fraction with a variable deceleration |
| `early_frac_contr` † | Fraction with an early deceleration |
| `any_dec_frac_contr` † | Fraction with any deceleration |
| `sinusoidal_min` | Minutes classified as sinusoidal |
| `uc_per_10min` † | Contractions per 10 min |
| `vlf` | log FHR power, 0–0.03 Hz (very low frequency) |
| `lf` | log FHR power, 0.03–0.15 Hz (low frequency) |
| `sd2` | Poincaré SD2 of successive 1-Hz FHR values (long-term variability; bpm) |
| `fhr_missing` | Fraction of missing FHR samples in the window |
| `uc_missing` † | Fraction of missing UC samples |
| `fhr_longest_gap_s` | Longest FHR gap (s) |
| `uc_longest_gap_s` † | Longest UC gap (s) |

† One of the 7 contraction variables removed in the leaner models (contraction rate, deceleration–contraction association and contraction-signal quality). Deceleration typing (early, late, variable, unclassified) still uses the contraction signal.

W60 uses the last 60 min of the available trace. All 552 public headers give `Sig2Birth = 0`. The database report allows recordings to end up to 30 min before birth, so exact alignment could not be established. The report describes 16 selected caesareans with pH ≤ 7.05 among 46 caesareans, whereas 15 of the 46 public headers meet that threshold. The study analysed 536 recordings, with 38 events at its primary cutoff of pH < 7.05.

## S4. Hyper-parameter search spaces

- CNN encoder (all learned models): width {16, 32, 48}; stages {3, 4}; kernel {5, 7, 9}; dropout {0.1, 0.3, 0.5}; learning rate {1e-3, 3e-4}; weight decay {1e-4, 1e-3, 1e-2}; batch 32; 6 random configurations; 3 inner folds; early stopping (≤ 60 epochs, patience 10).
- NeSy-E2E selection: inner outcome AUROC. Detectors: inner mean ACC/DEC event F1 (best grid threshold).
- Event thresholds: grid 0.10–0.90 (step 0.05), chosen on the 15% validation split by event F1; post-processing merges gaps ≤ 1 bin, minimum length 4 bins (ACC/DEC) or 8 bins (UC).
- Additive models (on extractor or detector variables): L2 logistic regression with cubic B-splines (quantile knots), C log-uniform on [10^−3^, 10^1^], knots {3, 4, 5}.
- Linear, HRV linear and FIGO-feature models: L2 logistic regression, C log-uniform on [10^−3^, 10^2^].
- HRV support-vector model: RBF support-vector machine, C log-uniform on [10^−2^, 10^2^], γ log-uniform on [10^−3^, 10^0^].
- All regression and support-vector models: 30 random configurations, 3 inner folds.

## S5. Event-level fidelity, full table (CTU-UHB, outer-test records, quality region)

In the primary analysis, we applied each fold's validation threshold to its test recordings. We pooled event counts within each repeat, then averaged F1, sensitivity and positive predictive value across the five repeats. The pre-specified analysis applied the median of the 50 fold thresholds to repeat-averaged frame probabilities. This threshold can depend indirectly on a test recording's labels (deviation 15 in S7). We used 2,000 paired record bootstrap resamples of the fixed out-of-fold predictions to estimate the 95% confidence intervals.

| Source | Event | Se | PPV | F1, fold-specific threshold (95% CI) | F1, pre-specified median threshold (95% CI) | Expert events |
|---|---|---|---|---|---|---|
| NeSy-E2E (end-to-end) | ACC | 0.365 | 0.214 | 0.269 (0.228–0.307) | 0.317 (0.266–0.364) | 529 |
| NeSy-E2E (end-to-end) | DEC | 0.617 | 0.478 | 0.538 (0.501–0.573) | 0.544 (0.506–0.582) | 1427 |
| NeSy-E2E (end-to-end) | UC | 0.709 | 0.802 | 0.752 (0.730–0.772) | 0.765 (0.743–0.786) | 3523 |
| Sequential detector | ACC | 0.538 | 0.383 | 0.447 (0.405–0.484) | 0.449 (0.407–0.488) | 529 |
| Sequential detector | DEC | 0.698 | 0.567 | 0.626 (0.593–0.654) | 0.627 (0.596–0.658) | 1427 |
| Sequential detector | UC | 0.796 | 0.833 | 0.814 (0.796–0.830) | 0.822 (0.804–0.838) | 3523 |
| Rule-guided detector | ACC | 0.601 | 0.410 | 0.487 (0.446–0.527) | 0.498 (0.455–0.539) | 529 |
| Rule-guided detector | DEC | 0.745 | 0.591 | 0.659 (0.627–0.687) | 0.664 (0.632–0.692) | 1427 |
| Rule-guided detector | UC | 0.791 | 0.822 | 0.806 (0.787–0.823) | 0.817 (0.799–0.834) | 3523 |
| Rule-based extractor | ACC | 0.698 | 0.359 | 0.474 (0.429–0.518) | 0.474 (0.429–0.517) | 529 |
| Rule-based extractor | DEC | 0.814 | 0.540 | 0.649 (0.614–0.680) | 0.649 (0.615–0.680) | 1427 |
| Rule-based extractor | UC | 0.930 | 0.639 | 0.757 (0.732–0.780) | 0.757 (0.735–0.779) | 3523 |
| NeSy-E2E without quality mask | ACC | 0.539 | 0.146 | — | 0.230 (0.194–0.265) | 529 |
| NeSy-E2E without quality mask | DEC | 0.547 | 0.467 | — | 0.504 (0.466–0.540) | 1427 |
| NeSy-E2E without quality mask | UC | 0.732 | 0.796 | — | 0.762 (0.738–0.783) | 3523 |

For NeSy-E2E without quality mask, Se and PPV are from the pre-specified median threshold; fold-specific outputs were not computed for this ablation.

**Pre-specified comparisons under both protocols** (difference in F1, learned source minus extractor; Holm adjustment within the pre-specified families):

| Test | Fold-specific: difference (95% CI), Holm p | Pre-specified median threshold: difference (95% CI), Holm p | Secondary, record-specific median of its own fold thresholds: difference (95% CI), unadjusted p |
|---|---|---|---|
| H1a, NeSy-E2E vs extractor, DEC | −0.111 (−0.137 to −0.088), < 0.003 | −0.105 (−0.132 to −0.079), < 0.003 | −0.100 (−0.128 to −0.074), < 0.001 |
| H1b, NeSy-E2E vs extractor, ACC | −0.205 (−0.247 to −0.161), < 0.003 | −0.157 (−0.209 to −0.106), < 0.003 | −0.192 (−0.241 to −0.143), < 0.001 |
| B1a, sequential detector vs extractor, DEC | −0.024 (−0.043 to −0.005), 0.036 | −0.022 (−0.042 to −0.002), 0.102 | −0.022 (−0.043 to −0.002), 0.028 |
| B1b, sequential detector vs extractor, ACC | −0.027 (−0.059 to 0.004), 0.154 | −0.025 (−0.064 to 0.018), 0.223 | −0.013 (−0.050 to 0.024), 0.473 |
| E1a, rule-guided detector vs extractor, DEC | +0.010 (−0.004 to 0.022), 0.264 | +0.014 (0.001 to 0.028), 0.038 | +0.017 (0.004 to 0.029), 0.018 |
| E1b, rule-guided detector vs extractor, ACC | +0.013 (−0.007 to 0.034), 0.264 | +0.024 (0.004 to 0.045), 0.038 | +0.010 (−0.012 to 0.031), 0.341 |

The secondary record-specific analysis was added after the fold-specific results were seen and is descriptive (deviation 18 in S7).

## S6. Full specification of the refitted linear model

L2 logistic regression on standardised variables, C = 0.0148 (median of the 50 fold-selected values); intercept (class-balanced fit) −0.6738; subtract log(pos_weight) = 2.5730 from the linear predictor. Missing values: median imputation with the values below.

| Variable | Imputation value | Mean | SD | Coefficient per SD (95% bootstrap CI) |
|---|---|---|---|---|
| `baseline` | 137.2 | 137.7 | 12.84 | +0.056 (−0.149 to 0.270) |
| `stv_ms` | 15.99 | 16.58 | 6.61 | +0.003 (−0.091 to 0.097) |
| `ltv_ms` | 104 | 108.5 | 44.84 | +0.120 (0.011 to 0.224) |
| `var_amp_median` | 15.28 | 16.06 | 5.507 | +0.151 (0.012 to 0.263) |
| `reduced_var_min` | 1.714 | 3.415 | 4.58 | −0.182 (−0.304 to −0.042) |
| `increased_var_min` | 14.65 | 15.51 | 8.826 | −0.002 (−0.176 to 0.152) |
| `n_acc` | 3 | 3.282 | 2.985 | +0.074 (−0.117 to 0.258) |
| `n_dec` | 6 | 6.763 | 4.451 | −0.081 (−0.169 to 0.015) |
| `n_dec_early` | 0 | 0.3918 | 0.6961 | +0.067 (−0.123 to 0.235) |
| `n_dec_late` | 0 | 0.4179 | 0.875 | −0.045 (−0.167 to 0.087) |
| `n_dec_variable` | 4 | 4.946 | 3.573 | −0.215 (−0.335 to −0.099) |
| `n_dec_prolonged` | 0 | 0.06343 | 0.2586 | +0.147 (−0.029 to 0.303) |
| `n_dec_other` | 1 | 0.944 | 1.199 | +0.301 (0.107 to 0.472) |
| `dec_total_min` | 5.473 | 6.17 | 4.302 | −0.019 (−0.109 to 0.076) |
| `dec_max_s` | 87.62 | 95.31 | 50.1 | +0.055 (−0.108 to 0.212) |
| `dec_area` | 8461 | 9786 | 7678 | −0.046 (−0.158 to 0.075) |
| `dec_max_depth` | 63.25 | 62.06 | 22.62 | −0.120 (−0.274 to 0.053) |
| `late_frac_contr` | 0 | 0.02842 | 0.05916 | +0.157 (−0.017 to 0.302) |
| `variable_frac_contr` | 0.1538 | 0.1784 | 0.154 | −0.085 (−0.217 to 0.049) |
| `early_frac_contr` | 0 | 0.02599 | 0.05146 | +0.054 (−0.115 to 0.195) |
| `any_dec_frac_contr` | 0.2143 | 0.2421 | 0.1895 | +0.063 (−0.053 to 0.166) |
| `sinusoidal_min` ‡ | 0 | 0 | 1 | +0.000 (0.000 to 0.000) |
| `uc_per_10min` | 2.833 | 2.766 | 1.073 | +0.021 (−0.165 to 0.190) |
| `vlf` | 4.783 | 4.709 | 0.7063 | +0.059 (−0.081 to 0.205) |
| `lf` | 3.179 | 3.1 | 0.6301 | −0.234 (−0.342 to −0.108) |
| `sd2` | 23.99 | 24.45 | 7.477 | +0.520 (0.305 to 0.705) |
| `fhr_missing` | 0.1361 | 0.1587 | 0.1251 | +0.268 (0.118 to 0.401) |
| `uc_missing` | 0.1977 | 0.2465 | 0.2127 | +0.055 (−0.102 to 0.205) |
| `fhr_longest_gap_s` | 128.2 | 306 | 362.1 | +0.165 (−0.002 to 0.317) |
| `uc_longest_gap_s` | 479.1 | 614.4 | 590 | +0.132 (−0.062 to 0.271) |

‡ Zero in all 536 recordings; the SD of 1 is the scaler's placeholder for a constant variable, so the term does not contribute.

## S7. Pre-specification record

- Plan 1 (end-to-end model, `docs/analiz_plani_v2.md`): frozen with git tag `analiz-plani-v2.0` in the development repository (2026-11-23T12:12:05+00:00) before any NeSy-E2E model was trained; the released file has the same SHA-256.
- Plan 2 and Addenda 1–4: SHA-256 checksums logged in `experiments/log.csv` before the runs they cover.

| Plan file | Role | Frozen (UTC) | SHA-256 |
|---|---|---|---|
| `docs/analiz_plani_v2.md` | Plan 1 (end-to-end model) | 2026-11-23T12:12:05+00:00 (git tag) | `3b9e4a9b7e6317e7…` |
| `docs/analiz_plani_v3.md` | Plan 2 (sequential detector; transparent models) | 2026-11-23T14:44:56+00:00 | `61e757b41aae3f94…` |
| `docs/analiz_plani_v3_ek1.md` | Addendum 1 (robustness settings) | 2026-11-23T14:53:16+00:00 | `d8cea70564ec8dc4…` |
| `docs/analiz_plani_v3_ek2.md` | Addendum 2 (rule-guided detector) | 2026-11-23T15:54:17+00:00 | `e11a3a24428fc65f…` |
| `docs/analiz_plani_v3_ek3.md` | Addendum 3 (distribution shifts) | 2026-11-23T19:27:11+00:00 | `b66aa019f46151d1…` |
| `docs/analiz_plani_v3_ek4.md` | Addendum 4 (distribution-shift fallbacks) | 2026-11-23T19:39:55+00:00 | `6aed4b74694dcf15…` |

**Deviations from the frozen plans** (listed with their timing in `docs/deviations.md` of the code repository):

Deviations 1–7 concern the benchmark plan (`docs/analiz_plani.md`, frozen with git tag `analiz-plani-v1.0`), under which the CNN ensemble, the FIGO-feature model and the weak-label NeSy comparator were developed.

| Log no. | Plan item | Deviation | Timing |
|---|---|---|---|
| 1 | Benchmark plan, quality index | Quality index defined as √(Se·Sp) instead of √(Se·PPV); the definition in the plan was an error. | Before any model result |
| 2 | Benchmark plan, calibration | Prior correction (logit minus the log of the positive-class weight) applied before temperature scaling, identically for all models. | Before any result |
| 3 | Benchmark plan, baseline concept | Four classes (< 100 bpm, 100–110 bpm, normal, tachycardia) instead of three, as required by the FIGO 2015 classification. | Before any result |
| 4 | Benchmark plan, concept list | Added concepts: any recurrent deceleration (`rep_any`), deceleration > 5 min (`very_prolonged`), prolonged increased variability and repetitive late or prolonged decelerations > 30 min. | Before any result |
| 5 | Benchmark plan, logic layer | Seven concept-level rules and a leak rule with learnable weights added to the pathological and suspicious class rules and combined by noisy-OR. | Before any result |
| 6 | Benchmark plan, sensitivity analyses | No nested re-tuning in the sensitivity analyses: each deep model used the configuration selected most often in the primary analysis; the FIGO-feature model kept full nested tuning. | Before the sensitivity results; after the primary results of the baseline models |
| 7 | Benchmark plan, data | Expert annotation sets (Romagnoli et al., Hruban et al., FHRMA) obtained for Plan 1; not used in the benchmark analyses. | After the benchmark results, which did not change |
| 8 | Plan 1, hypothesis H1c | Exploratory diagnostic check of the chance-level acceleration concept (index and record alignment, per-fold AUROC). No model, threshold or primary result was changed. | After the Plan 1 results |
| 9 | Plan 1, interpretation rule of the intervention test | The rule (a small change means that the outcome flows through the concepts) assumed concepts close to the expert reference. With concepts near chance, a small change was not interpreted as evidence that the risk flows through the expert-referenced concepts. Numbers unchanged. | After the Plan 1 results |
| 10 | Plan 2, tuning budget (6 random configurations) | The scikit-learn models (additive and linear models, the FIGO-feature model, the HRV models and the additive models on detector concepts) were tuned with 30 configurations, as in the benchmark protocol released with the code; 6 applies to the deep networks. | While writing the code, before any result of this study |
| 11 | Plan 2, list of measures | Added descriptive reporting: sensitivity at 85% and 90% specificity, calibration slope and intercept, decision-curve net benefit and the ΔAUROC of each transparent model against the ensemble. No tests. | After the primary results |
| 12 | Plan 2, scope | Added TRIPOD+AI descriptive items: cohort characteristics, numbers per analysis, subgroup AUROCs and the full specification of the refitted linear model. Clinical metadata were used for description only. | After all results |
| 13 | Code (rule-based extractor) | Correction of a coding error: when overlapping decelerations were merged, their areas were summed, so overlapping samples were counted twice. The merged event's area is now computed once over the merged span. Every model using the FIGO variables was re-run with the same protocol; detector, NeSy and CNN outputs were unaffected. | After all results |
| 14 | Addendum 1, outcome "pH < 7.15" | The label actually used was the benchmark's secondary outcome pH ≤ 7.15 (106 events); the text of the addendum contained a typing error. Reported as run; not re-run with < 7.15. | After all results |
| 15 | Plans 1 and 2, event threshold for outer-test records (median of the 50 fold thresholds applied to repeat-averaged probabilities) | Main analysis changed to fold-specific thresholds: each outer-test record is thresholded with its own fold's validation-selected threshold; F1 per repeat, averaged over repeats; pre-specified comparisons recomputed with the same bootstrap and Holm families. The pooled median threshold depends indirectly on a record's own labels through other folds' validation splits. The choice of the main analysis was logged before its results were seen; the pre-specified results are reported alongside. | After the pre-specified results; before the fold-specific results |
| 16 | Plan 2, scope | Added descriptive analyses: a linear model of the three HRV variables, HRV and signal-quality ablations of the linear model, and bootstrap CIs for the calibration slope and intercept. No tests. | After all results |
| 17 | Plan 1, intervention test | Added a check of how well the approximate rebuild reproduces the recorded risks. | After all results |
| 18 | Plans 1 and 2, event threshold and complementarity | Added two descriptive analyses: a record-specific threshold (the median of the thresholds of the folds in which the record was an outer-test record) and the average of the linear model and the CNN ensemble alongside the pre-specified average of the additive model and the ensemble. | After the fold-specific and primary results |
| 19 | Scope (post hoc) | Added member-level concept substitution and a controlled outcome-loss ablation. The original analyses are reported alongside. | After the original results |
| 20 | Scope (post hoc) | Added validation-selected operating points, a single-feature FHR signal loss comparator, and paired net-benefit intervals on the original out-of-fold predictions. | After the original results |
| 21 | Scope (post hoc) | Added paired AUROC comparisons of three saved logistic models (HRV; HRV and FIGO-derived variables; all 30 variables) on identical folds (S17). No model was refitted. | After all results |

## S8. Controlled ablation of acidaemia outcome loss

We fitted one member per arm in each fold of a ten-fold repeat, covering 536 test recordings. Both arms used width 16, 3 stages, kernel 7, dropout 0.3, AdamW, learning rate 0.001, weight decay 0.001, batch size 32 and 24 fixed epochs. They shared the inputs, supervised concept losses, seed, minibatch order and fitting records. The acidaemia loss had weight 0 in one arm and 1 in the other. We selected thresholds separately on the same validation records and estimated intervals using 2,000 paired record bootstrap resamples with the fitted models held fixed.

| Event match | Event | F1 without outcome loss | F1 with outcome loss | Difference (95% CI) |
|---|---|---|---|---|
| Any overlap | ACC | 0.267 | 0.272 | −0.005 (−0.041 to 0.030) |
| Any overlap | DEC | 0.560 | 0.518 | +0.042 (0.021 to 0.063) |
| IoU ≥ 0.5 | ACC | 0.165 | 0.114 | +0.051 (0.017 to 0.085) |
| IoU ≥ 0.5 | DEC | 0.414 | 0.356 | +0.058 (0.032 to 0.084) |

## S9. Calibration, operating points and continuous pH (primary analysis)

| Model | Cal. slope | Cal. intercept | Sens. at 85% spec. | Sens. at 90% spec. |
|---|---|---|---|---|
| CNN ensemble | 1.27 | −0.05 | 0.50 | 0.42 |
| FIGO-feature model | 0.78 | −0.28 | 0.50 | 0.42 |
| HRV support-vector model | 0.91 | −0.05 | 0.61 | 0.58 |
| Linear model | 1.25 | 0.09 | 0.61 | 0.50 |
| Additive model | 1.13 | 0.02 | 0.55 | 0.53 |
| HRV linear model (post hoc) | 1.13 | 0.05 | 0.63 | 0.61 |

On each outer-training validation subset, the CNN, additive and logistic models used temperature scaling after class-prior correction. The HRV support-vector model used Platt scaling of the uncorrected decision scores. The calibration and operating-point summaries in the table were added after the plan was frozen (deviation 11 in S7).

**Continuous pH.** Harrell's C used the repeat-averaged out-of-fold risks. It gives credit for assigning higher risk to the lower-pH member of each pair with different pH, with half credit for a tied risk. The estimates were 0.587 (CNN ensemble), 0.550 (FIGO-feature model), 0.601 (HRV support-vector model), 0.596 (Linear model), 0.584 (Additive model).

## S10. Rule weights of the end-to-end model (medians over 50 outer folds; 2.5–97.5th percentiles)

| Rule | Median | 2.5% | 97.5% |
|---|---|---|---|
| Baseline risk (`leak`) | 0.066 | 0.047 | 0.081 |
| FIGO pathological pattern (`pathological`) | 0.135 | 0.110 | 0.164 |
| FIGO suspicious pattern (`suspicious`) | 0.203 | 0.135 | 0.252 |
| Recurrent late decelerations (`rep_late`) | 0.197 | 0.119 | 0.334 |
| Recurrent variable decelerations (`rep_variable`) | 0.144 | 0.112 | 0.241 |
| Prolonged deceleration (`prolonged`) | 0.234 | 0.136 | 0.327 |
| Fetal tachycardia (`tachycardia`) | 0.218 | 0.139 | 0.290 |
| Absence of accelerations (`no_accel`) | 0.172 | 0.130 | 0.205 |
| Reduced variability (`reduced_variability`) | 0.218 | 0.139 | 0.300 |
| Recurrent decelerations (`rep_any`) | 0.225 | 0.135 | 0.291 |

## S11. Per-repeat event F1 (fold-specific thresholds)

| Source | Event | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 |
|---|---|---|---|---|---|---|
| NeSy-E2E (end-to-end) | ACC | 0.291 | 0.271 | 0.275 | 0.250 | 0.258 |
| NeSy-E2E (end-to-end) | DEC | 0.551 | 0.552 | 0.522 | 0.528 | 0.537 |
| Sequential detector | ACC | 0.448 | 0.455 | 0.442 | 0.439 | 0.450 |
| Sequential detector | DEC | 0.625 | 0.623 | 0.630 | 0.628 | 0.621 |
| Rule-guided detector | ACC | 0.490 | 0.491 | 0.492 | 0.482 | 0.481 |
| Rule-guided detector | DEC | 0.659 | 0.659 | 0.657 | 0.659 | 0.661 |

## S12. Deceleration-type confusion among matched decelerations (exploratory; pre-specified median threshold)

**NeSy-E2E** — matched pairs 832, macro-F1 0.375; recall E 0.69, L 0.00, V 0.86, P 0.00

| Expert type (rows) / predicted type (columns) | E | V |
|---|---|---|
| E | 233 | 104 |
| L | 32 | 13 |
| P | 6 | 10 |
| V | 59 | 375 |

**Rule-based extractor** — matched pairs 1162, macro-F1 0.294; recall E 0.11, L 0.13, V 0.70, P 0.26

| Expert type (rows) / predicted type (columns) | E | L | P | V | U |
|---|---|---|---|---|---|
| E | 54 | 52 | 6 | 339 | 26 |
| L | 6 | 11 | 2 | 56 | 10 |
| P | 2 | 3 | 5 | 5 | 4 |
| V | 26 | 24 | 5 | 408 | 118 |

E, early; L, late; P, prolonged; V, variable; U, unclassified deceleration (see S2).

## S13. Stricter matching (IoU ≥ 0.5)

| Source | ACC F1 | DEC F1 | UC F1 |
|---|---|---|---|
| NeSy-E2E (end-to-end) | 0.141 | 0.389 | 0.646 |
| Sequential detector | 0.297 | 0.488 | 0.732 |
| Rule-guided detector | 0.404 | 0.555 | 0.726 |
| Rule-based extractor | 0.439 | 0.574 | 0.609 |

Fold-specific thresholds, IoU ≥ 0.5. Rule-guided detector minus extractor: ACC −0.035 (−0.060 to −0.011); DEC −0.019 (−0.039 to −0.001); UC +0.117 (0.098 to 0.136).

## S14. External expert consensus (FHRMA, 66 public recordings)

| Source | ACC Se | ACC PPV | ACC F1 | DEC Se | DEC PPV | DEC F1 |
|---|---|---|---|---|---|---|
| NeSy-E2E | 0.135 | 0.248 | 0.175 | 0.236 | 0.677 | 0.350 |
| Sequential detector | 0.207 | 0.502 | 0.293 | 0.388 | 0.736 | 0.508 |
| Rule-guided detector | 0.344 | 0.536 | 0.419 | 0.460 | 0.735 | 0.566 |
| Rule-based extractor | 0.349 | 0.650 | 0.454 | 0.492 | 0.796 | 0.608 |

## S15. Window concepts against expert-derived labels (n = 340 annotated, quality-eligible records)

| Concept | Positives | NeSy-E2E AUROC (κ) | Weak-label NeSy AUROC (κ) | Weak-label NeSy without concept supervision AUROC | Rule-based weak label κ |
|---|---|---|---|---|---|
| Accelerations present (`accel`) | 190 | 0.514 (0.02) | 0.516 (0.07) | 0.517 | 0.23 |
| Recurrent decelerations (>50% of contractions) (`rep_any`) | 26 | 0.539 (−0.05) | 0.605 (0.09) | 0.577 | 0.45 |
| Recurrent late decelerations (`rep_late`) | 3 | not evaluated (< 20 positives) | | | |
| Recurrent variable decelerations (`rep_variable`) | 1 | not evaluated (< 20 positives) | | | |
| Early decelerations (`early`) | 182 | 0.616 (0.16) | 0.574 (0.00) | 0.462 | 0.26 |
| Prolonged deceleration (>3 min) (`prolonged`) | 19 | not evaluated (< 20 positives) | | | |
| Deceleration >5 min (`very_prolonged`) | 6 | not evaluated (< 20 positives) | | | |
| Repetitive late/prolonged decelerations >30 min (`rep_late_long`) | 4 | not evaluated (< 20 positives) | | | |

Weak-label NeSy without concept supervision: the benchmark ablation with the weak-label concept loss switched off (weight 0).

## S16. Concept intervention test (340 recordings, 16 events)

**Original approximate intervention (fold models rebuilt from member-averaged logits and rule weights; pre-specified)**

| Model | AUROC rebuilt, no intervention | AUROC intervened | Log-loss rebuilt | Log-loss intervened | Mean \|Δ risk\| |
|---|---|---|---|---|---|
| NeSy-E2E | 0.690 | 0.688 | 0.188 | 0.187 | 0.024 |
| Weak-label NeSy | 0.690 | 0.693 | 0.208 | 0.196 | 0.026 |

Fidelity of the rebuild (deviation 17): without intervention, the rebuilt risks agreed with the recorded model risks with NeSy-E2E Spearman 0.98, mean absolute difference 0.024, AUROC 0.690 rebuilt vs 0.699 recorded; Weak-label NeSy Spearman 0.89, mean absolute difference 0.056, AUROC 0.690 rebuilt vs 0.678 recorded. The rebuild error is of the same size as the approximate intervention effect, so the approximate analysis can exclude only a large dependence.

**Member-level intervention (post hoc)**

We reproduced the predictions of 250 saved NeSy-E2E members before replacing their concepts. The mean absolute difference from the saved predictions was 9.39e-06, and the pooled baseline AUROC was 0.699. We inserted expert binary labels as logits of ±9.2, applied each member's own calibration and averaged the resulting probabilities. We estimated 95% intervals for AUROC differences using 2,000 paired outcome-stratified record resamples.

| Substitution | AUROC after | ΔAUROC (95% CI) | Mean / median absolute change (pp) | Absolute change ≥ 5 pp |
|---|---|---|---|---|
| All eight expert-referenced concepts | 0.706 | +0.007 (−0.075 to 0.098) | 2.29 / 1.50 | 44/340 |
| Acceleration only | 0.699 | +0.001 (−0.055 to 0.054) | 1.23 / 1.26 | 0/340 |
| Deceleration family | 0.697 | −0.001 (−0.057 to 0.073) | 1.82 / 0.91 | 33/340 |

pp, percentage points. The five-point cutoff describes the distribution. Results for the weak-label comparator are available for the approximate intervention.

## S17. Variable-group ablations and complementarity (primary analysis)

- Full additive model minus additive model without HRV: +0.057 (0.001 to 0.114).
- Full additive model minus additive model without signal quality: +0.014 (−0.031 to 0.055).
- Full linear model minus linear model without HRV (deviation 16): +0.050 (−0.006 to 0.107); minus linear model without signal quality: +0.005 (−0.036 to 0.050).
- Linear model without the 7 contraction variables: AUROC 0.809 (0.727–0.876); additive model without them: 0.776 (0.691–0.848).
- Average of additive model and CNN ensemble (pre-specified, descriptive): AUROC 0.812; average of linear model and CNN ensemble (added after the results, deviation 18 in S7): 0.818.

**Exploratory predictor-group comparison.** All three logistic models used the same 536 CTU-UHB recordings (38 events). We checked that test and validation membership and outcome labels matched for every outer fold. We calculated AUROC after averaging the five held-out probabilities per record. For pairwise differences, we used 2,000 paired record bootstrap resamples of the saved predictions (seed 20261123). The saved model runs were `hrv_lr_W60`, `transp_lr_noq_W60` and `transp_lr_W60`. Signal quality may also affect descriptors outside the explicit quality group.

| Predictor blocks | Inputs | AUROC (95% CI) | ΔAUROC vs preceding row (95% CI) |
|:---|:---:|:---:|:---:|
| HRV | 3 | 0.782 (0.680 to 0.873) | Reference |
| HRV and FIGO-derived variables | 26 | 0.803 (0.714 to 0.879) | +0.020 (−0.043 to +0.085) |
| HRV, FIGO-derived and explicit quality variables | 30 | 0.808 (0.720 to 0.880) | +0.005 (−0.036 to +0.050) |

The 30-variable model minus HRV alone gave ΔAUROC +0.026 (95% CI −0.039 to +0.096).

The supplementary reproducibility files include `paired_predictor_blocks.py` and its aggregate output. The fold prediction arrays are produced by the three named model runs in `scripts/reproduce_paper.sh` of the code repository. After generating those arrays, run `python paired_predictor_blocks.py <repository-root> predictor_blocks.json` in the repository's Python environment.

## S18. Net benefit at selected thresholds (primary analysis)

| Threshold | Treat all | CNN ensemble | FIGO-feature model | HRV support-vector model | Linear model | Additive model |
|---|---|---|---|---|---|---|
| 0.05 | 0.0220 | 0.0359 | 0.0299 | 0.0399 | 0.0404 | 0.0388 |
| 0.1 | −0.0323 | 0.0226 | 0.0164 | 0.0292 | 0.0245 | 0.0230 |
| 0.15 | −0.0931 | 0.0136 | 0.0123 | 0.0246 | 0.0217 | 0.0237 |
| 0.2 | −0.1614 | 0.0098 | 0.0117 | 0.0187 | 0.0126 | 0.0121 |
| 0.25 | −0.2388 | 0.0068 | 0.0087 | 0.0187 | 0.0100 | 0.0087 |
| 0.3 | −0.3273 | 0.0027 | 0.0085 | 0.0101 | 0.0091 | 0.0059 |

## S19. Validation-selected thresholds and paired net benefit

For each fold, the decision threshold was the empirical 90th percentile of negative predictions in its validation split. The values below pool test decisions over the five repeats. They describe the performance of the threshold-selection procedure.

| Model | Test sensitivity | Test specificity |
|---|---|---|
| CNN ensemble | 0.374 | 0.885 |
| Additive model | 0.442 | 0.902 |
| Linear model | 0.474 | 0.910 |
| HRV linear model (post hoc) | 0.574 | 0.904 |
| FIGO-feature model | 0.353 | 0.908 |

FHR loss alone had AUROC 0.707 as a descriptive ranking score.

| Hypothetical threshold | Comparison | Net benefit difference (95% CI) |
|---|---|---|
| 0.10 | Linear minus CNN ensemble | +0.0019 (−0.0100 to 0.0139) |
| 0.10 | Linear minus HRV linear model (post hoc) | −0.0066 (−0.0195 to 0.0046) |
| 0.20 | Linear minus CNN ensemble | +0.0028 (−0.0098 to 0.0154) |
| 0.20 | Linear minus HRV linear model (post hoc) | −0.0065 (−0.0182 to 0.0033) |

Intervals are from 2,000 paired record bootstrap draws on fixed out-of-fold probabilities. The estimates use the CTU-UHB sample event rate and are descriptive of this selected cohort.

## S20. Robustness settings (AUROC, 95% CI; single-run AUROC mean ± sample SD over 5 repeats)

| Setting | n / events | CNN ensemble | FIGO-feature model | Linear model | Additive model | Single-run linear | Single-run CNN |
|---|---|---|---|---|---|---|---|
| Primary (last 60 min) | 536 / 38 | 0.794 (0.718–0.867) | 0.725 (0.637–0.811) | 0.808 (0.720–0.880) | 0.792 (0.709–0.865) | 0.783 ± 0.010 | 0.745 ± 0.048 |
| Last 30 min | 536 / 38 | 0.806 (0.732–0.872) | 0.555 (0.440–0.666) | 0.788 (0.701–0.862) | 0.796 (0.716–0.861) | 0.771 ± 0.025 | 0.785 ± 0.017 |
| Last 90 min | 536 / 38 | 0.769 (0.686–0.848) | 0.689 (0.595–0.771) | 0.776 (0.678–0.862) | 0.782 (0.683–0.864) | 0.753 ± 0.014 | 0.736 ± 0.026 |
| First stage, last 60 min | 536 / 38 | 0.778 (0.699–0.850) | 0.700 (0.591–0.795) | 0.746 (0.638–0.838) | 0.752 (0.665–0.830) | 0.731 ± 0.018 | 0.757 ± 0.011 |
| All 552 recordings | 552 / 40 | 0.790 (0.708–0.862) | 0.709 (0.620–0.793) | 0.792 (0.706–0.870) | 0.797 (0.716–0.867) | 0.772 ± 0.017 | 0.767 ± 0.026 |
| pH ≤ 7.15 | 536 / 106 | 0.723 (0.668–0.776) | 0.656 (0.598–0.716) | 0.698 (0.638–0.758) | 0.682 (0.620–0.745) | 0.688 ± 0.014 | 0.706 ± 0.013 |
| BDecf ≥ 12 mmol/L | 526 / 16 | 0.763 (0.612–0.893) | 0.704 (0.525–0.867) | 0.732 (0.577–0.862) | 0.676 (0.492–0.830) | 0.708 ± 0.052 | 0.739 ± 0.025 |
| 5-min Apgar < 7 | 536 / 19 | 0.361 (0.221–0.508) | 0.394 (0.253–0.541) | 0.415 (0.242–0.586) | 0.510 (0.397–0.625) | 0.401 ± 0.073 | 0.397 ± 0.038 |

In the seven additional settings the transparent and FIGO-feature models were fully re-tuned; as specified in Addendum 1, the CNN ensemble used the configuration selected most often in the primary analysis (deviation 6 in S7).

## S21. Subgroup discrimination (AUROC, 95% CI; descriptive)

| Subgroup | n / events | CNN ensemble | Linear model | Additive model |
|---|---|---|---|---|
| Nulliparous | 368 / 33 | 0.783 (0.701–0.857) | 0.796 (0.708–0.873) | 0.779 (0.689–0.853) |
| Parous | 168 / 5 | 0.822 (0.494–1.000) | 0.817 (0.477–0.988) | 0.804 (0.408–0.991) |
| Maternal age < 35 y | 463 / 29 | 0.824 (0.742–0.891) | 0.834 (0.738–0.913) | 0.820 (0.729–0.897) |
| Maternal age ≥ 35 y | 73 / 9 | 0.694 (0.479–0.888) | 0.729 (0.530–0.906) | 0.708 (0.496–0.883) |
| Gestation < 41 wk | 336 / 26 | 0.792 (0.687–0.881) | 0.785 (0.674–0.878) | 0.767 (0.654–0.861) |
| Gestation ≥ 41 wk | 200 / 12 | 0.797 (0.682–0.894) | 0.857 (0.739–0.951) | 0.849 (0.737–0.943) |
| Induced labour | 208 / 13 | 0.767 (0.619–0.899) | 0.780 (0.615–0.922) | 0.766 (0.604–0.902) |
| Spontaneous labour | 328 / 25 | 0.812 (0.718–0.889) | 0.823 (0.729–0.906) | 0.806 (0.703–0.894) |
| Vaginal delivery | 498 / 25 | 0.759 (0.657–0.850) | 0.792 (0.680–0.891) | 0.778 (0.668–0.871) |
| Caesarean delivery | 38 / 13 | 0.717 (0.546–0.872) | 0.634 (0.443–0.825) | 0.671 (0.456–0.881) |

Descriptive TRIPOD+AI addition (deviation 12 in S7).

## S22. Distribution shifts: recalibration within the target group (leave-one-out)

Shift A: train on vaginal births (498 recordings, 25 events), test on caesareans (38, 13 events). Shift B: train on FHR loss ≤ median (268, 9 events), test on the rest (268, 29 events). UC, uterine contractions. Some features still require UC timing for deceleration typing.

| Shift | Model | AUROC | Intercept raw | Brier raw | Brier, intercept update | Slope after intercept update | Brier, intercept + slope |
|---|---|---|---|---|---|---|---|
| A | Linear model | 0.735 | +1.98 | 0.2734 | 0.2110 | 0.66 | 0.2230 |
| A | Additive model | 0.665 | +1.63 | 0.2532 | 0.2239 | 0.48 | 0.2286 |
| A | FIGO-feature model | 0.665 | +1.55 | 0.2571 | 0.2199 | 0.50 | 0.2317 |
| A | HRV support-vector model | 0.720 | +2.20 | 0.2661 | 0.2078 | 0.54 | 0.2164 |
| A | CNN ensemble | 0.797 | +1.88 | 0.2661 | 0.1926 | 1.04 | 0.2004 |
| A | Linear, no UC variables | 0.726 | — | 0.2739 | 0.2136 | 0.56 | 0.2235 |
| A | Additive, no UC variables | 0.686 | — | 0.2617 | 0.2152 | 0.59 | 0.2253 |
| B | Linear model | 0.664 | −0.22 | 0.1067 | 0.1049 | 0.36 | 0.0938 |
| B | Additive model | 0.638 | +1.34 | 0.0983 | 0.0938 | 0.66 | 0.0944 |
| B | FIGO-feature model | 0.589 | +1.37 | 0.0998 | 0.0962 | 0.48 | 0.0961 |
| B | HRV support-vector model | 0.709 | +1.17 | 0.0876 | 0.0911 | 0.43 | 0.0861 |
| B | CNN ensemble | 0.708 | +1.81 | 0.1029 | 0.0931 | 1.23 | 0.0942 |
| B | Linear, no UC variables | 0.654 | — | 0.0990 | 0.1003 | 0.37 | 0.0933 |
| B | Additive, no UC variables | 0.665 | — | 0.0948 | 0.0928 | 0.65 | 0.0929 |

Paired AUROC differences from the CNN ensemble (95% CI, 2,000 paired bootstrap resamples): Shift A, linear model −0.062 (−0.218 to 0.087), additive model −0.132 (−0.332 to 0.054); Shift B, linear model −0.044 (−0.163 to 0.065), additive model −0.070 (−0.204 to 0.057).

## S23. Reference-standard audit

Spearman correlation of expert ACC+DEC density with FHR loss in the 60-min window: −0.49.

| FHR-loss tertile | n | Acidaemic | Median density, pH < 7.05 (events/h) | Median density, pH ≥ 7.05 (events/h) | Mann–Whitney p |
|---|---|---|---|---|---|
| low | 171 | 6 | 4.0 | 8.0 | 0.25 |
| mid | 171 | 10 | 4.0 | 3.0 | 0.87 |
| high | 171 | 20 | 1.0 | 1.0 | 0.98 |

Recordings failing the record-level quality criterion: 20/36 acidaemic vs 153/477 non-acidaemic annotated recordings. Mean share of the window excluded from expert supervision: 0.64 vs 0.42.

Annotation inventory: 10,208 annotated episodes (UC 7,027, DEC 2,261, ACC 823, tachycardia 54, bradycardia 43).

Exploratory diagnostic check of the window acceleration concept (deviation 8 in S7; concept results in S15): per-fold AUROC 0.522 (SD 0.101) over 50 folds.
