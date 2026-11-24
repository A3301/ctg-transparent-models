"""Event post-processing and event-level matching on 4-s bins (docs/analiz_plani_v2.md §4.2, §6.1).

Intervals are in 4 Hz window samples; bins are 16 samples (4 s).
"""
from __future__ import annotations

import numpy as np

from src.eval.concept_fidelity import match

BIN = 16
MIN_BINS = {0: 4, 1: 4, 2: 8}          # ACC, DEC: 16 s; UC: 32 s
MERGE_GAP = 1                           # merge runs separated by <= 1 bin
THR_GRID = np.round(np.arange(0.10, 0.901, 0.05), 2)


def bins_to_events(p, thr, min_bins, merge_gap=MERGE_GAP):
    """p: (n_bins,) probabilities -> list of (start_sample, end_sample)."""
    pos = np.r_[False, p >= thr, False].astype(np.int8)
    d = np.diff(pos)
    runs = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s * BIN, e * BIN) for s, e in merged if e - s >= min_bins]


def in_region(ev, region):
    """Keep events with >= 50 % of their duration inside the bin mask `region`."""
    out = []
    for s, e in ev:
        b0, b1 = s // BIN, max(s // BIN + 1, -(-e // BIN))
        if region[b0:b1].mean() >= 0.5:
            out.append((s, e))
    return out


def event_counts(ref, det, region, iou_min=0.0):
    ref, det = in_region(ref, region), in_region(det, region)
    return len(match(ref, det, iou_min)), len(ref), len(det)


def f1(tp, nr, nd):
    return 2 * tp / (nr + nd) if (nr + nd) else np.nan


def expert_events(T, i, label):
    m = (T["ev_rec"] == i) & (T["ev_label"] == label)
    return list(zip(T["ev_start"][m].tolist(), T["ev_end"][m].tolist()))


def expert_events_typed(T, i):
    m = (T["ev_rec"] == i) & (T["ev_label"] == 1)
    return list(zip(T["ev_start"][m].tolist(), T["ev_end"][m].tolist(), T["ev_type"][m].tolist()))


def select_thresholds(probs, rec_idx, T, region_key="q"):
    """probs: (n, 3, n_bins) frame probabilities for validation records (indices into T rows).
    Returns {label: threshold} maximising pooled event F1 in the region; plus the F1 curves."""
    thr, curves = {}, {}
    for lab in (0, 1, 2):
        best, best_t, curve = -1.0, 0.5, []
        for t in THR_GRID:
            tp = nr = nd = 0
            for k, i in enumerate(rec_idx):
                det = bins_to_events(probs[k, lab], t, MIN_BINS[lab])
                a, b, c = event_counts(expert_events(T, i, lab), det, T[region_key][i].astype(float))
                tp, nr, nd = tp + a, nr + b, nd + c
            v = f1(tp, nr, nd)
            curve.append(float(v) if np.isfinite(v) else None)
            if np.isfinite(v) and v > best:
                best, best_t = v, float(t)
        thr[lab], curves[lab] = best_t, curve
    return thr, curves
