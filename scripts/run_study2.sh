#!/usr/bin/env bash
# End-to-end model (docs/analiz_plani_v2.md): NeSy-v2 with nested tuning, then the pre-specified no-mask ablation
# (reuses NeSy-v2's per-fold configuration). Resumable: finished folds are skipped.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
W=${WORKERS:-6}
export PYTHONIOENCODING=utf-8
[ -f data/processed/expert_targets_W60.npz ] || $PY -m src.data.concept_targets
run() { exp=$1; shift; mkdir -p results/cv/$exp; $PY -m src.run.cv --exp $exp "$@" --workers $W >> results/cv/$exp/log.txt 2>&1; echo "$(date +%T) finished $exp"; }
echo "$(date +%T) start end-to-end model runs"
run nesy2_W60        --model nesy2        --ensemble 5
run nesy2nomask_W60  --model nesy2_nomask --ensemble 5 --fixed_cfg_from nesy2_W60
echo "$(date +%T) NESY2_DONE"
