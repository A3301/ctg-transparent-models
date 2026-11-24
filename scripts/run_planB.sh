#!/usr/bin/env bash
# Plan B (docs/analiz_plani_v3.md §4): expert-supervised event detector, 5x10 outer folds, resumable.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
echo "$(date +%T) start planB"
$PY -m src.run.cv --model det --exp det_W60 --ensemble 5 --workers 6 >> results/cv/det_W60.log 2>&1
echo "$(date +%T) finished det_W60"
echo "PLANB_DET_DONE"
$PY -m src.run.stage2 --workers 8 >> results/cv/transpdet_gam_W60.log 2>&1
echo "$(date +%T) finished transpdet_gam_W60"
echo "PLANB_DONE"
