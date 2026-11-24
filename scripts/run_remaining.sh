#!/usr/bin/env bash
# Resume everything (resumable: finished folds are skipped). Intended to run detached from the terminal.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
LOG=results/cv/_chain.log
echo "$(date '+%F %T') chain start" >> $LOG
bash scripts/fill_missing.sh 6 >> $LOG 2>&1
mkdir -p results/cv/consist_georgoulas
$PY -m src.run.cv --exp consist_georgoulas --model georgoulas --window S1_30 --label y_ph705le --splits consist_georgoulas_5x10_all552 --workers 8 >> results/cv/consist_georgoulas/log.txt 2>&1
echo "$(date '+%F %T') consist_georgoulas done" >> $LOG
WORKERS=6 bash scripts/run_phase45.sh >> $LOG 2>&1
bash scripts/fill_missing.sh 4 >> $LOG 2>&1
echo "$(date '+%F %T') CHAIN_DONE" >> $LOG
