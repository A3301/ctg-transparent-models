#!/usr/bin/env bash
# Evaluation chain after scripts/run_fix13.sh (deviation log #13–#17). Aggregate results only.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
$PY -m src.eval.study3_eval            # C1/B2 non-inferiority, C2/B1 superiority, discrimination, shape functions
$PY -m src.eval.study3_planD           # eight settings
$PY -m src.eval.study3_planE           # rule-guided detector (unchanged inputs; re-run for consistency)
$PY -m src.eval.study3_extra           # calibration, net benefit, descriptive additions (#16), calibration / decision-curve figure
$PY -m src.eval.study3_tripod          # cohort, subgroups, refitted final model
$PY -m src.eval.study3_transport_eval && $PY -m src.eval.study3_transport_eval --ek4
$PY -m src.eval.foldwise_events        # fold-specific event fidelity (uses the new C2 for the plan-2 Holm family)
$PY -m src.eval.intervention_check
$PY -m src.eval.fix13_impact
$PY -m src.eval.study2_figures && $PY -m src.eval.study3_figures
echo EVAL_FIX13_DONE
