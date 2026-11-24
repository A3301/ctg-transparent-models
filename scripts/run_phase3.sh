#!/usr/bin/env bash
# Phase 3: baselines under the frozen nested-CV protocol (resumable).
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
run() { exp=$1; shift; mkdir -p results/cv/$exp; $PY -m src.run.cv --exp $exp "$@" >> results/cv/$exp/log.txt 2>&1; echo "$(date +%T) finished $exp"; }
run figo_lr_W60     --model figo_lr    --workers 8
run figo_xgb_W60    --model figo_xgb   --workers 8
run georgoulas_W60  --model georgoulas --workers 8
run cnn_W60         --model cnn        --workers 6
run cnn_ens_W60     --model cnn        --workers 6 --ensemble 5
run transformer_W60 --model transformer --workers 6
run gru_W60         --model gru        --workers 6
run mcnn_W60        --model mcnn       --workers 6
run fhrlinet_W60    --model fhrlinet   --workers 6
# consistency checks under the original papers' conditions (no exclusion, 552 records)
run consist_fhrlinet --model fhrlinet --splits consist_mendis_5x5_all552 --folds 0,1,2,3,4 --workers 6
run consist_georgoulas --model georgoulas --window S1_30 --label y_ph705le --splits consist_georgoulas_5x10_all552 --workers 8
echo ALL_DONE
