#!/usr/bin/env bash
# Deviation log #13 (deceleration-area fix) and #16 (descriptive additions): rerun every experiment that uses the
# rule-extractor FIGO features, with the original commands (scripts/run_parallel_b.sh, run_planD.sh, transport.py).
# Old outputs are moved to results/cv/_prefix13/ and results/transport/_prefix13/ for the impact comparison.
# CPU only (scikit-learn); CNN, HRV-SVM, NeSy and detector runs are not affected and are not rerun.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
LOG=results/cv/_fix13.log
mkdir -p results/cv/_prefix13 results/transport/_prefix13

park() { [ -d "results/cv/$1" ] && mv "results/cv/$1" "results/cv/_prefix13/$1"; }
run() {  # exp model window label splits
  park "$1"
  $PY -m src.run.cv --exp "$1" --model "$2" --window "$3" --label "$4" --splits "$5" --workers 8 >> "results/cv/_fix13_$2.log" 2>&1
  echo "$(date '+%F %T') $1 $(ls results/cv/$1/r*_f*.npz 2>/dev/null | wc -l) folds" >> $LOG
}
settings() {  # model
  run "$1_W60"     "$1" W60   y_ph705      outer_splits
  run "$1_ph715"   "$1" W60   y_ph715      outer_splits_ph715
  run "$1_bdecf12" "$1" W60   y_bdecf12    outer_splits_bdecf12
  run "$1_apgar5"  "$1" W60   y_apgar5_lt7 outer_splits_apgar5
  run "$1_all552"  "$1" W60   y_ph705      outer_splits_all552
  run "$1_W30"     "$1" W30   y_ph705      outer_splits
  run "$1_W90"     "$1" W90   y_ph705      outer_splits
  run "$1_S1_60"   "$1" S1_60 y_ph705      outer_splits
}
echo "$(date '+%F %T') FIX13 start" >> $LOG
( settings figo_lr ) &
( settings transp_lr
  for m in transp_lr_nouc transp_lr_nohrv transp_lr_noq hrv_lr; do run "${m}_W60" "$m" W60 y_ph705 outer_splits; done ) &
( settings transp_gam
  for m in transp_gam_noq transp_gam_nohrv transp_gam_nouc; do run "${m}_W60" "$m" W60 y_ph705 outer_splits; done ) &
wait
echo "$(date '+%F %T') CV_DONE" >> $LOG
for s in C1 C3; do for m in transp_lr transp_gam figo_lr transp_lr_nouc transp_gam_nouc; do
  [ -f "results/transport/${s}_${m}.npz" ] && mv "results/transport/${s}_${m}.npz" results/transport/_prefix13/
done; done
$PY -m src.run.transport --models transp_lr,transp_gam,figo_lr,transp_lr_nouc,transp_gam_nouc >> results/transport_run.log 2>&1
echo "$(date '+%F %T') FIX13_DONE" >> $LOG
