# Deviations from the pre-specified analysis plans

Expert-referenced interpretability in intrapartum cardiotocography models for fetal acidaemia — Akdağlı, Yanık, Yıldız.

Source: Supplementary material, section S7. Numbers match `experiments/log.csv`.

**Deviations from the frozen plans** (with timing):

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
