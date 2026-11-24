#!/usr/bin/env bash
# Plan E (docs/analiz_plani_v3_ek2.md): rule-guided detector + stage 2. Resumable.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
echo "$(date +%T) start planE"
$PY -m src.run.cv --model detk --exp detk_W60 --ensemble 5 --workers 6 >> results/cv/detk_W60.log 2>&1
echo "$(date +%T) finished detk_W60"
$PY -m src.run.stage2 --workers 8 --exp transpdetk_gam --det detk_W60 >> results/cv/transpdetk_gam_W60.log 2>&1
echo "$(date +%T) finished transpdetk_gam_W60"
echo "PLANE_DONE"
