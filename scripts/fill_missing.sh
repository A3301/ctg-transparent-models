#!/usr/bin/env bash
# Re-run any experiment whose folds are missing (e.g. after CUDA OOM). Resumable: done folds are skipped.
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
for d in results/cv/*/; do
  e=$(basename $d); [ -f $d/run_args.json ] || continue
  n=$(ls $d | grep -c "^r.*npz$")
  want=$($PY -c "import json;a=json.load(open('$d/run_args.json'));print(len(a['repeats'].split(','))*len(a['folds'].split(',')))")
  if [ "$n" -lt "$want" ]; then
    args=$($PY -c "import json;a=json.load(open('$d/run_args.json'));a['workers']=${1:-4};print(' '.join(f'--{k} {v}' for k,v in a.items() if v is not None))")
    echo "$(date +%T) filling $e ($n/$want)"; $PY -m src.run.cv $args >> $d/log.txt 2>&1
  fi
done
echo FILL_DONE
