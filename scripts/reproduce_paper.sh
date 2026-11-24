#!/usr/bin/env bash
# Reproduce the analyses of this paper after the benchmark pipeline (scripts/reproduce_all.sh).
# GPU required; several hours.
# Order follows the pre-specification sequence (docs/analiz_plani_v2.md, v3, ek1-ek4); deviations #13–#21
# (docs/deviations.md) are covered: fixed FIGO features, fold-specific event thresholds,
# descriptive additions and the post hoc analyses. scripts/run_fix13.sh + run_eval_fix13.sh document the
# rerun of the models affected by the deceleration-area fix (#13).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
$PY -m src.eval.power_pilot                            # pilot power check used to plan v2 (planning only; first run before #13)
$PY src/data/fetch_annotations.py                      # expert annotations (verify licences first)
$PY -m src.data.annotations                            # Romagnoli events (local only, not redistributed)
$PY -m src.eval.concept_fidelity                       # rule-extractor fidelity
bash scripts/run_study2.sh                             # end-to-end NeSy model + ablation (plan v2)
$PY -m src.eval.study2_eval && $PY -m src.eval.study2_fhrma
$PY src/eval/study2_diag_h1c.py                        # exploratory H1c check, printed only (deviation #8)
for m in transp_gam transp_lr transp_gam_noq transp_gam_nohrv transp_lr_nouc transp_gam_nouc \
         transp_lr_nohrv transp_lr_noq hrv_lr; do       # last three: deviation #16
  $PY -m src.run.cv --model $m --exp ${m}_W60 --workers 4
done
bash scripts/run_planB.sh                              # sequential detector + stage 2 (plan v3)
bash scripts/run_planD.sh                              # robustness settings (ek1)
bash scripts/run_planE.sh                              # rule-guided detector + stage 2 (ek2)
$PY -m src.run.transport                               # distribution shifts (ek3)
$PY -m src.run.transport --models transp_lr_nouc,transp_gam_nouc
$PY -m src.eval.study3_eval && $PY -m src.eval.study3_planD && $PY -m src.eval.study3_planE
$PY -m src.eval.study3_extra && $PY -m src.eval.study3_tripod
$PY -m src.eval.study3_transport_eval && $PY -m src.eval.study3_transport_eval --ek4
$PY -m src.eval.foldwise_events                        # fold-specific event thresholds (deviation #15)
$PY -m src.eval.intervention_check                     # check of the approximate intervention test (#17)
if [ -d results/cv/_prefix13 ]; then                   # old vs new results of the deceleration-area fix (#13);
  $PY -m src.eval.fix13_impact                         # needs the pre-fix outputs parked by scripts/run_fix13.sh
fi
$PY -m src.eval.posthoc_member_intervention            # post hoc: member-level concept substitution (#19)
$PY -m src.eval.posthoc_controlled_loss                # post hoc: controlled outcome-loss ablation, GPU (#19)
$PY -m src.eval.posthoc_transparent                    # post hoc: operating points, FHR-loss comparator, net-benefit CIs (#20)
$PY manuscript/reproducibility/paired_predictor_blocks.py . manuscript/reproducibility/predictor_blocks.json   # post hoc (#21)
$PY -m src.eval.study2_figures && $PY -m src.eval.study3_figures && $PY -m src.eval.manuscript_figures
echo "PAPER_REPRODUCED"
