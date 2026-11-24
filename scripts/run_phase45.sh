#!/usr/bin/env bash
# Phase 4/5: ablations (reuse proposed configs), sensitivity analyses, flagged clinical-prior experiment.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
W=${WORKERS:-4}
run() { exp=$1; shift; mkdir -p results/cv/$exp; $PY -m src.run.cv --exp $exp "$@" --workers $W >> results/cv/$exp/log.txt 2>&1; echo "$(date +%T) finished $exp"; }
# --- ablations (no re-tuning: configs of nesy_ens_W60 per outer fold)
run nesy_single_W60    --model nesy           --ensemble 1 --fixed_cfg_from nesy_ens_W60
run nesy_nologic_W60   --model nesy_nologic   --ensemble 5 --fixed_cfg_from nesy_ens_W60
run nesy_nocsup_W60    --model nesy_nocsup    --ensemble 5 --fixed_cfg_from nesy_ens_W60
run nesy_rulesonly_W60 --model nesy_rulesonly --ensemble 1 --fixed_cfg_from nesy_ens_W60
# --- sensitivity analyses (full nested protocol) for NeSy-Bayes, CNN-Bayes, FIGO-LR
for m in nesy:5 cnn:5 figo_lr:1; do
  mod=${m%%:*}; ens=${m##*:}; tag=$mod; [ "$ens" = "5" ] && tag=${mod}_ens
  FX="--fixed_cfg_from mode:${tag}_W60"   # plan disi #6: no nested re-tuning in sensitivity analyses
  [ "$mod" = "figo_lr" ] && FX=""         # cheap + continuous C -> keep full nested tuning
  run ${tag}_S1_60 $FX   --model $mod --ensemble $ens --window S1_60
  run ${tag}_W30 $FX     --model $mod --ensemble $ens --window W30
  run ${tag}_W90 $FX     --model $mod --ensemble $ens --window W90
  run ${tag}_ph715 $FX   --model $mod --ensemble $ens --label y_ph715      --splits outer_splits_ph715
  run ${tag}_bdecf12 $FX --model $mod --ensemble $ens --label y_bdecf12    --splits outer_splits_bdecf12
  run ${tag}_apgar5 $FX  --model $mod --ensemble $ens --label y_apgar5_lt7 --splits outer_splits_apgar5
  run ${tag}_all552 $FX  --model $mod --ensemble $ens --splits outer_splits_all552
done
# --- FLAGGED exploratory: clinical priors as logical facts
run nesyclin_ens_W60 --model nesy_clin --ensemble 5 --fixed_cfg_from mode:nesy_ens_W60
echo PHASE45_DONE
