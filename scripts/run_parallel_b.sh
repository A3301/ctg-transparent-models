#!/usr/bin/env bash
# Parallel chain B: FIGO-LR sensitivity (CPU) + flagged clinical-prior experiment (GPU).
# The main chain reaches these last; finished folds are skipped there (resumable).
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
LOG=results/cv/_chain_b.log
run() { exp=$1; shift; mkdir -p results/cv/$exp; $PY -m src.run.cv --exp $exp "$@" >> results/cv/$exp/log.txt 2>&1; echo "$(date +%T) finished $exp" >> $LOG; }
echo "$(date '+%F %T') chain B start" >> $LOG
run nesyclin_ens_W60 --model nesy_clin --ensemble 5 --fixed_cfg_from mode:nesy_ens_W60 --workers 4 &
run figo_lr_S1_60   --model figo_lr --window S1_60 --workers 8
run figo_lr_W30     --model figo_lr --window W30 --workers 8
run figo_lr_W90     --model figo_lr --window W90 --workers 8
run figo_lr_ph715   --model figo_lr --label y_ph715 --splits outer_splits_ph715 --workers 8
run figo_lr_bdecf12 --model figo_lr --label y_bdecf12 --splits outer_splits_bdecf12 --workers 8
run figo_lr_apgar5  --model figo_lr --label y_apgar5_lt7 --splits outer_splits_apgar5 --workers 8
run figo_lr_all552  --model figo_lr --splits outer_splits_all552 --workers 8
wait
echo "$(date '+%F %T') CHAIN_B_DONE" >> $LOG
