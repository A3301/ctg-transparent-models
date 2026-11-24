"""Sensitivity-analysis table (Phase 5): windows, labels, no-exclusion, clinical priors.

Each row = one experiment summarised with src.eval.summarize (repeat-averaged OOF, bootstrap CIs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.compare import get_summary  # noqa: E402

SETTINGS = [("W60 (birincil)", "W60"), ("S1_60 (yalnız I. evre)", "S1_60"), ("W30", "W30"), ("W90", "W90"),
            ("pH ≤ 7,15", "ph715"), ("BDecf ≥ 12", "bdecf12"), ("Apgar5 < 7", "apgar5"),
            ("Dışlama yok (552)", "all552")]
MODELS = [("NeSy-Bayes", "nesy_ens"), ("CNN-Bayes", "cnn_ens"), ("FIGO-LR", "figo_lr")]


def fmt(v):
    return f"{v[0]:.3f} ({v[1]:.3f}–{v[2]:.3f})"


def main():
    rows = []
    for sname, suf in SETTINGS:
        for mname, pre in MODELS:
            exp = f"{pre}_{suf}"
            d = ROOT / "results" / "cv" / exp
            if not d.exists() or len(list(d.glob("r*_f*.npz"))) < 50:
                rows.append({"Ayar": sname, "Model": mname, "n/n+": "eksik"})
                continue
            s = get_summary(exp)
            rows.append({"Ayar": sname, "Model": mname, "n/n+": f"{s['n_records']}/{s['n_pos']}",
                         "AUROC": fmt(s["auroc"]), "AUPRC": fmt(s["auprc"]),
                         "Se@Sp90": f"{s['sens_at_spec90'][0]:.3f}", "Brier": f"{s['brier'][0]:.3f}"})
    clin = ROOT / "results" / "cv" / "nesyclin_ens_W60"
    if clin.exists() and len(list(clin.glob("r*_f*.npz"))) == 50:
        s = get_summary("nesyclin_ens_W60")
        rows.append({"Ayar": "W60 + klinik ön bilgi (KEŞİFSEL)", "Model": "NeSy-Bayes + klinik",
                     "n/n+": f"{s['n_records']}/{s['n_pos']}", "AUROC": fmt(s["auroc"]), "AUPRC": fmt(s["auprc"]),
                     "Se@Sp90": f"{s['sens_at_spec90'][0]:.3f}", "Brier": f"{s['brier'][0]:.3f}"})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    df.to_csv(out / "sensitivity.csv", index=False)
    (out / "sensitivity.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
