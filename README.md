# Expert-referenced interpretability in intrapartum cardiotocography models for fetal acidaemia

Code, pre-specified analysis plans, run log, aggregate results and the released transparent model for the paper that evaluates whether intrapartum cardiotocography (CTG) models that discriminate fetal acidaemia well also agree with held-out (CTU-UHB) and external (FHRMA) expert event annotations. On CTU-UHB, an end-to-end neuro-symbolic model with a probabilistic FIGO 2015 logic layer, separately trained and rule-guided event detectors, a rule-based FIGO extractor, transparent additive and linear models of deterministic FIGO, heart-rate-variability and signal-quality variables, and a deep CNN ensemble are compared under one record-level nested cross-validation protocol, with expert events from CTU-UHB and the external FHRMA consensus set.

Manuscript submitted to *Artificial Intelligence in Medicine*. Authors: Ali Akdağlı, Hüseyin Yanık, Ali Yıldız (Mersin University, Türkiye).
Archived version: https://doi.org/10.5281/zenodo.24491162

## Contents

| Path | Content |
|---|---|
| `src/` | Data cleaning, features, models, nested cross-validation, evaluation, figures |
| `scripts/` | `reproduce_all.sh`: benchmark pipeline (cleaning, splits, baseline models); `reproduce_paper.sh`: this paper's analyses |
| `docs/` | Pre-specified analysis plans (Turkish) and the list of deviations (`deviations.md`) |
| `experiments/log.csv` | Every run with configuration, seed, metrics and the SHA-256 checksums of the plans logged before the runs |
| `results/`, `figures/` | Aggregate results (JSON/CSV) and figures, plus record-level tables derived from the openly licensed CTU-UHB database (e.g. `results/eda/metadata_all.csv`, `results/features/`); no raw signals and no record-level tables derived from the CC BY-NC-ND expert annotations |
| `manuscript/` | Supplementary material of the submitted manuscript (tables taken from the result files) and the script and output of the predictor-group comparison (`manuscript/reproducibility/`) |

`results/study2/k3_and_diag_numbers.json` and `results/annotations/fidelity_ctu_by_quality_posthoc.json` were written by interactive post hoc checks; no script for them is included.

## Pre-specified analysis plans

`docs/analiz_plani.md` is the frozen benchmark plan (version 1.0), under which the CNN ensemble, the FIGO-feature model and the weak-label NeSy comparator were developed. The analysis plans for this paper are `docs/analiz_plani_v2.md` (Plan 1, end-to-end model), `docs/analiz_plani_v3.md` (Plan 2, sequential detector and transparent models) and its addenda `docs/analiz_plani_v3_ek1.md` (robustness), `docs/analiz_plani_v3_ek2.md` (rule-guided detector), `docs/analiz_plani_v3_ek3.md` and `docs/analiz_plani_v3_ek4.md` (transportability). The SHA-256 checksums of Plan 2 and its addenda were written to `experiments/log.csv` before the corresponding models were trained. Plan 1 was frozen by a git tag in the development repository (commit 8d953f6, 2026-11-23 15:12 +03:00) before the end-to-end model was trained, so its checksum is given here rather than in `experiments/log.csv`: its SHA-256 at that tag is 3b9e4a9b7e6317e71aaffb8ed0e938ce7313c2cedefc3790464aeab57ce2eb47, identical to the file in this repository. Recompute the checksums with `sha256sum docs/analiz_plani*.md` to check. Deviations from the frozen plans, with their timing, are listed in `docs/deviations.md` (Supplementary S7); analyses added after the primary results are labelled post hoc.

## Data

Raw signals and expert annotations are not redistributed.
- **CTU-UHB** (PhysioNet, doi:10.13026/C22013): download `ctu-uhb-ctgdb-1.0.0.zip` from PhysioNet, save it as `data/raw/ctu-uhb-ctgdb_1.0.0.zip` and save PhysioNet's checksum list (`SHA256SUMS.txt`) as `data/raw/SHA256SUMS_provider.txt`; `src/data/verify_and_extract.py` checks the download against it.
- **Expert annotations**: `python src/data/fetch_annotations.py` downloads them and verifies their checksums.
  The Romagnoli et al. (2020) annotations are licensed CC BY-NC-ND 4.0; files and record-level tables derived from them are not included.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux/macOS (scripts pick the right interpreter; override with PY=...)
bash scripts/reproduce_all.sh      # benchmark pipeline (cleaning, splits, baseline models)
bash scripts/reproduce_paper.sh    # analyses of this paper (GPU, several hours)
```

Patient-level splits are fixed in `results/splits/`; tuning uses inner cross-validation only.

Figures: `python -m src.eval.manuscript_figures` redraws Figures 1, 2 and 4 from the shipped results. Figures 3, 5 and 6 and the predictor-group comparison (`manuscript/reproducibility/`) also need the expert annotations and the fold-level prediction arrays, which `scripts/reproduce_all.sh` and `scripts/reproduce_paper.sh` download or recreate; missing inputs are reported and skipped.

## Apply the released transparent model

```bash
python -m src.predict.transparent rec1.csv rec2.csv --out predictions.csv
```

Input: one CSV per recording with columns `fhr` (bpm) and `uc`, 4 Hz, ending at delivery or at the time point of interest.
The tool applies the study's cleaning, computes the 30 variables on the last 60 min and applies the linear model
refitted on all 536 recordings (`results/study3/transp_lr_final_model.json`; same prior correction, no temperature
scaling). Its discrimination is estimated by the cross-validated performance of the same procedure. The model is not
externally validated and not intended for clinical use; recalibrate it on local outcome data before interpreting
absolute risks.

## Licence

- Code: MIT (`LICENSE`).
- Documentation, analysis plans, figures and aggregate results: CC BY 4.0 (`LICENSE-docs`).
- Third-party data and annotations remain under their own licences and are not included.

## Citation

See `CITATION.cff`. Repository: https://github.com/A3301/ctg-transparent-models
