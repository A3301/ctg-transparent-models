#!/usr/bin/env bash
# Benchmark pipeline (cleaning, splits, baseline models) from the raw PhysioNet zip with one command.
#   Prerequisites: Python 3.12, NVIDIA GPU (CUDA 12.8 build of PyTorch), ~12 GPU-hours.
#   Place ctu-uhb-ctgdb_1.0.0.zip (SHA-256 50619d82...07525) and SHA256SUMS.txt
#   (renamed SHA256SUMS_provider.txt) in data/raw/ first.
# Usage: bash scripts/reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
  python -m venv .venv
  .venv/Scripts/python -m pip install -r requirements.txt
fi
PY=${PY:-.venv/Scripts/python}; [ -e "$PY" ] || [ -e "$PY.exe" ] || PY=.venv/bin/python   # Windows or Linux/macOS virtual environment
$PY src/data/verify_and_extract.py          # integrity check + extraction (Phase 0)
$PY src/eda/phase0_eda.py                   # EDA, labels (Phase 0)
$PY src/preprocess/build_dataset.py         # cleaning, windows, exclusion (Phase 2)
$PY src/eval/splits.py                      # frozen CV splits (Phase 2)
$PY - <<'EOF'
# consistency-check splits (Phase 3), identical to the original run
import sys, json; sys.path.insert(0, '.')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from src.eval.splits import load_population, SEED
def mk(label, k, reps):
    d = load_population(True); d = d[d[label].notna()]; recs = d.index.values; y = d[label].astype(int).values
    out = {'label': label, 'seed': SEED, 'n_records': int(len(d)), 'repeats': []}
    for r in range(reps):
        skf = StratifiedKFold(k, shuffle=True, random_state=SEED + r)
        out['repeats'].append([{'train': recs[a].tolist(), 'test': recs[b].tolist()} for a, b in skf.split(recs, y)])
    return out
Path('results/splits/consist_mendis_5x5_all552.json').write_text(json.dumps(mk('y_ph705', 5, 5)))
Path('results/splits/consist_georgoulas_5x10_all552.json').write_text(json.dumps(mk('y_ph705le', 10, 5)))
EOF
$PY src/features/build_features.py          # FIGO features + weak concept labels (Phase 3)
$PY src/features/build_hrv.py               # HRV features (Phase 3)
bash scripts/run_phase3.sh                  # baselines + consistency checks
$PY -m src.run.cv --exp nesy_ens_W60 --model nesy --ensemble 5 --workers 6   # weak-label NeSy ensemble (Phase 4)
WORKERS=6 bash scripts/run_phase45.sh       # ablations, sensitivity, clinical-prior experiment
bash scripts/fill_missing.sh 4              # re-run any fold lost to CUDA OOM
$PY -m src.eval.compare                     # main table, H1/H2, family tests, figures (Phase 5)
$PY -m src.eval.sensitivity                 # sensitivity table
$PY src/eval/summarize.py consist_fhrlinet consist_georgoulas
$PY src/eval/explain.py                     # rule weights, concept fidelity
echo "Reproduction finished. Tables: results/tables/  Figures: figures/results/"
