"""Phase 0 exploratory data analysis for CTU-UHB.

Outputs: results/eda/*.csv|json and figures/eda/*.png
Descriptive only: no modelling, no tuning, no thresholds chosen from outcomes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.ctu_uhb import FS, load_metadata, load_signals, make_labels  # noqa: E402

SEED = 20261122
OUT = ROOT / "results" / "eda"
FIG = ROOT / "figures" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
FHR_LO, FHR_HI = 50.0, 210.0


def signal_stats(rec: str, meta_row: pd.Series) -> dict:
    fhr, uc = load_signals(rec)
    n = len(fhr)
    fhr_nan = ~np.isfinite(fhr)
    fhr_zero = np.isfinite(fhr) & (fhr == 0)
    fhr_oor = np.isfinite(fhr) & (fhr != 0) & ((fhr < FHR_LO) | (fhr > FHR_HI))
    fhr_bad = fhr_nan | fhr_zero | fhr_oor
    uc_bad = ~np.isfinite(uc) | (uc == 0)

    def frac_last(mask, minutes):
        k = int(minutes * 60 * FS)
        return float(mask[-k:].mean()) if n >= k else float("nan")

    # Longest gap of invalid FHR (seconds)
    longest = 0
    run = 0
    for b in fhr_bad:
        run = run + 1 if b else 0
        longest = max(longest, run)

    pos2 = meta_row.get("pos_stage2", np.nan)
    has_s2 = bool(np.isfinite(pos2) and 0 < pos2 < n)
    good = fhr[~fhr_bad]
    return {
        "record": rec,
        "n_samples_sig": n,
        "fhr_nan_frac": float(fhr_nan.mean()),
        "fhr_zero_frac": float(fhr_zero.mean()),
        "fhr_oor_frac": float(fhr_oor.mean()),
        "fhr_loss_frac": float(fhr_bad.mean()),
        "fhr_loss_last30": frac_last(fhr_bad, 30),
        "fhr_loss_last60": frac_last(fhr_bad, 60),
        "fhr_longest_gap_s": longest / FS,
        "uc_loss_frac": float(uc_bad.mean()),
        "fhr_median_valid": float(np.median(good)) if good.size else np.nan,
        "has_stage2_in_signal": has_s2,
        "stage2_in_signal_min": float((n - pos2) / FS / 60) if has_s2 else 0.0,
    }


def plot_record(rec: str, meta: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    fhr, uc = load_signals(rec)
    t = np.arange(len(fhr)) / FS / 60.0
    fhr_p = fhr.copy()
    fhr_p[(fhr_p == 0) | ~np.isfinite(fhr_p)] = np.nan
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(t, fhr_p, lw=0.5, color="tab:blue")
    ax[0].axhspan(110, 160, color="green", alpha=0.07, label="110-160 bpm")
    ax[0].set_ylim(50, 210)
    ax[0].set_ylabel("FHR (bpm)")
    ax[1].plot(t, uc, lw=0.5, color="tab:orange")
    ax[1].set_ylabel("UC")
    ax[1].set_xlabel("Time from recording start (min)")
    pos2 = meta.loc[rec, "pos_stage2"]
    if np.isfinite(pos2) and 0 < pos2 < len(fhr):
        for a in ax:
            a.axvline(pos2 / FS / 60, color="red", ls="--", lw=1)
        ax[0].text(pos2 / FS / 60, 205, " stage II", color="red", va="top", fontsize=8)
    ph = meta.loc[rec, "ph"]
    ax[0].set_title(f"Record {rec}  |  pH={ph:.2f}  |  "
                    f"pH<7.05: {int(labels.loc[rec, 'y_ph705'])}  |  "
                    f"Sig2Birth={meta.loc[rec, 'sig2birth']:.0f}")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    meta = load_metadata()
    labels = make_labels(meta)
    meta.to_csv(OUT / "metadata_all.csv")
    labels.to_csv(OUT / "labels.csv")

    stats = pd.DataFrame([signal_stats(r, meta.loc[r]) for r in meta.index]).set_index("record")
    stats.to_csv(OUT / "signal_quality.csv")
    df = meta.join(stats).join(labels)

    summary: dict = {"n_records": int(len(df))}
    # header n_samples vs signal length
    summary["header_vs_signal_len_mismatch"] = int((df["n_samples"] != df["n_samples_sig"]).sum())
    # Durations
    summary["duration_min"] = df["duration_min"].describe().round(2).to_dict()
    summary["n_duration_ge_60"] = int((df["duration_min"] >= 60).sum())
    summary["n_duration_ge_90"] = int((df["duration_min"] >= 89.99).sum())
    summary["n_duration_lt_30"] = int((df["duration_min"] < 30).sum())
    # Signal loss
    summary["fhr_loss_frac"] = df["fhr_loss_frac"].describe().round(4).to_dict()
    summary["fhr_loss_last60"] = df["fhr_loss_last60"].describe().round(4).to_dict()
    summary["fhr_zero_frac_mean"] = round(float(df["fhr_zero_frac"].mean()), 4)
    summary["fhr_oor_frac_mean"] = round(float(df["fhr_oor_frac"].mean()), 5)
    summary["fhr_nan_frac_mean"] = round(float(df["fhr_nan_frac"].mean()), 5)
    for thr in (0.2, 0.3, 0.4, 0.5):
        summary[f"n_fhr_loss_gt_{thr}"] = int((df["fhr_loss_frac"] > thr).sum())
        summary[f"n_fhr_loss_last60_gt_{thr}"] = int((df["fhr_loss_last60"] > thr).sum())
    summary["uc_loss_frac"] = df["uc_loss_frac"].describe().round(4).to_dict()
    summary["n_uc_loss_gt_0.5"] = int((df["uc_loss_frac"] > 0.5).sum())
    # Stage II, record type, Sig2Birth
    summary["n_has_stage2_in_signal"] = int(df["has_stage2_in_signal"].sum())
    summary["stage2_in_signal_min"] = df.loc[df.has_stage2_in_signal, "stage2_in_signal_min"].describe().round(2).to_dict()
    summary["pos_stage2_value_counts_special"] = {
        "zero": int((df["pos_stage2"] == 0).sum()),
        "ge_n_samples": int((df["pos_stage2"] >= df["n_samples"]).sum()),
        "nan": int(df["pos_stage2"].isna().sum()),
    }
    summary["rec_type_counts"] = df["rec_type"].value_counts().sort_index().to_dict()
    summary["sig2birth"] = df["sig2birth"].describe().round(2).to_dict()
    summary["n_sig2birth_gt0"] = int((df["sig2birth"] > 0).sum())
    summary["deliv_type_counts"] = df["deliv_type"].value_counts().sort_index().to_dict()
    # Consistency check for Sig2Birth units:
    # stage2_min (header) vs signal-stage-II minutes + sig2birth (assumed minutes)
    m = df.has_stage2_in_signal
    diff = df.loc[m, "stage2_min"] - (df.loc[m, "stage2_in_signal_min"] + df.loc[m, "sig2birth"])
    summary["stage2_consistency_diff_min"] = diff.describe().round(2).to_dict()
    summary["stage2_consistency_abs_le_2min_frac"] = round(float((diff.abs() <= 2).mean()), 3)
    # Labels
    summary["missing_outcomes"] = {c: int(df[c].isna().sum()) for c in ["ph", "bdecf", "apgar5", "be", "pco2"]}
    summary["labels_positive"] = {c: int(labels[c].sum()) for c in labels.columns}
    summary["labels_n_defined"] = {c: int(labels[c].notna().sum()) for c in labels.columns}
    summary["labels_prevalence"] = {c: round(float(labels[c].mean()), 4) for c in labels.columns}
    summary["ph"] = df["ph"].describe().round(3).to_dict()
    summary["label_overlap_ph705_vs_bdecf12"] = pd.crosstab(
        df["y_ph705"], df["y_bdecf12"]).to_dict()
    summary["positives_by_stage2"] = pd.crosstab(
        df["has_stage2_in_signal"], df["y_ph705"]).to_dict()
    summary["positives_by_rec_type"] = pd.crosstab(df["rec_type"], df["y_ph705"]).to_dict()

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        return o

    (OUT / "eda_summary.json").write_text(json.dumps(_clean(summary), indent=2))

    # ---------- Figures ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["ph"].dropna(), bins=50, color="tab:blue", edgecolor="white")
    for x, c, lab in [(7.05, "red", "7.05 (primary)"), (7.15, "orange", "7.15")]:
        ax.axvline(x, color=c, ls="--", label=lab)
    ax.set_xlabel("Umbilical artery pH")
    ax.set_ylabel("Records")
    ax.set_title(f"pH distribution (n={df['ph'].notna().sum()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "ph_histogram.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].hist(df["bdecf"].dropna(), bins=40, color="tab:purple", edgecolor="white")
    axs[0].axvline(12, color="red", ls="--")
    axs[0].set_xlabel("BDecf (mmol/L)")
    axs[1].hist(df["apgar5"].dropna(), bins=np.arange(-0.5, 11), color="tab:green", edgecolor="white")
    axs[1].axvline(6.5, color="red", ls="--")
    axs[1].set_xlabel("Apgar 5 min")
    lab_names = {"y_ph705": "pH<7.05", "y_ph715": "pH≤7.15", "y_bdecf12": "BDecf≥12", "y_apgar5_lt7": "Apgar5<7"}
    pos = [labels[c].sum() for c in lab_names]
    axs[2].bar(list(lab_names.values()), pos, color="tab:red")
    for i, p in enumerate(pos):
        axs[2].text(i, p, f"{int(p)}", ha="center", va="bottom")
    axs[2].set_ylabel("Positive records")
    fig.tight_layout()
    fig.savefig(FIG / "label_distributions.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].hist(df["duration_min"], bins=40, color="tab:gray", edgecolor="white")
    axs[0].set_xlabel("Recording duration (min)")
    axs[1].hist(df["fhr_loss_frac"] * 100, bins=40, color="tab:blue", edgecolor="white")
    axs[1].set_xlabel("FHR signal loss, whole record (%)")
    axs[2].hist(df["fhr_loss_last60"].dropna() * 100, bins=40, color="tab:cyan", edgecolor="white")
    axs[2].set_xlabel("FHR signal loss, last 60 min (%)")
    for a in axs:
        a.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(FIG / "duration_and_signal_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for y, c in [(0, "tab:blue"), (1, "tab:red")]:
        s = df.loc[df["y_ph705"] == y, "fhr_loss_frac"] * 100
        ax.hist(s, bins=30, alpha=0.6, density=True, color=c, label=f"pH<7.05 = {y} (n={len(s)})")
    ax.set_xlabel("FHR signal loss (%)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Signal loss by primary label (descriptive)")
    fig.tight_layout()
    fig.savefig(FIG / "signal_loss_by_label.png", dpi=150)
    plt.close(fig)

    # Example records: 3 random negatives + 3 random positives (fixed seed), with stage II if possible
    rng = np.random.default_rng(SEED)
    picks = []
    for y in (0, 1):
        ids = df.index[df["y_ph705"] == y].tolist()
        picks += list(rng.choice(ids, size=3, replace=False))
    for rec in picks:
        plot_record(rec, meta, labels, FIG / f"example_{rec}.png")
    summary_picks = {"example_records": [str(p) for p in picks], "seed": SEED}
    (OUT / "eda_examples.json").write_text(json.dumps(summary_picks, indent=2))
    print(json.dumps(_clean(summary), indent=2))


if __name__ == "__main__":
    main()
