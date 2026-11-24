#!/usr/bin/env bash
# Plan D (docs/analiz_plani_v3_ek1.md): transparent models across the benchmark sensitivity settings.
# CPU only (sklearn). Settings mirror the benchmark figo_lr sensitivity runs (window, label, splits).
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
run() {  # model suffix window label splits
  $PY -m src.run.cv --model "$1" --exp "$1_$2" --window "$3" --label "$4" --splits "$5" --workers 4 \
    >> results/cv/_planD.log 2>&1
  echo "$(date +%T) $1_$2 $(ls results/cv/$1_$2/r*_f*.npz 2>/dev/null | wc -l) folds"
}
for m in transp_lr transp_gam; do
  run $m ph715   W60   y_ph715      outer_splits_ph715
  run $m bdecf12 W60   y_bdecf12    outer_splits_bdecf12
  run $m apgar5  W60   y_apgar5_lt7 outer_splits_apgar5
  run $m all552  W60   y_ph705      outer_splits_all552
  run $m W30     W30   y_ph705      outer_splits
  run $m W90     W90   y_ph705      outer_splits
  run $m S1_60   S1_60 y_ph705      outer_splits
done
echo "PLAND_DONE"
