"""FHR / UC cleaning for CTU-UHB (4 Hz).

All thresholds are fixed here, documented in docs/faz_2_rapor.md, and chosen
without looking at outcome labels.

FHR
  1. Values == 0 or outside [50, 210] bpm -> missing.                (protocol, common practice)
  2. Valid samples are split into runs at adjacent jumps > 25 bpm.     (protocol; Phase 2 check: only
     0.03 % of adjacent differences exceed 25 bpm, so true physiology is rarely cut)
     Runs shorter than 5 s are treated as spikes -> missing.
     (A "level rule" that also removed runs deviating > 25 bpm from the surrounding +/- 60 s
     median was tried and REJECTED: visual check (figures/preprocess/check_level_rule.png)
     showed it deleted true deep decelerations. It is kept behind USE_LEVEL_RULE = False
UC_FLAT_WIN_S = 60.0
UC_FLAT_RANGE = 1.0.)
  3. Gaps <= 15 s between valid samples are linearly interpolated;     (protocol)
     longer gaps stay missing and are flagged by the mask.
UC
  1. NaN, negative values, and runs of exact 0 lasting >= 10 s -> missing (sensor off).
     Shorter zero runs are kept as true low values (UC 25th percentile is 1.5 units).
     Flat line: samples whose centred 60 s window has range (max - min) <= 1 unit, in runs
     lasting >= 60 s -> missing (non-working transducer; seen in records 1451, 2003).
  2. Gaps <= 15 s interpolated; longer gaps masked.
  3. Values are kept in device units; per-fold normalisation is done inside model pipelines.
"""
from __future__ import annotations

import numpy as np

FS = 4
FHR_LO, FHR_HI = 50.0, 210.0
JUMP_BPM = 25.0
MIN_RUN_S = 5.0
CONTEXT_S = 60.0
MAX_INTERP_S = 15.0
UC_ZERO_RUN_S = 10.0
USE_LEVEL_RULE = False
UC_FLAT_WIN_S = 60.0
UC_FLAT_RANGE = 1.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def interpolate_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN gaps of length <= max_gap that are bounded on both sides.
    Returns (filled, valid_mask) where valid_mask marks observed or interpolated samples."""
    y = x.copy()
    nan = ~np.isfinite(y)
    for s, e in _runs(nan):
        if s == 0 or e == len(y) or (e - s) > max_gap:
            continue
        y[s:e] = np.linspace(y[s - 1], y[e], e - s + 2)[1:-1]
    return y, np.isfinite(y)


def clean_fhr(fhr: np.ndarray) -> dict:
    x = fhr.astype(float).copy()
    n = len(x)
    stats = {}
    bad_range = ~np.isfinite(x) | (x == 0) | (x < FHR_LO) | (x > FHR_HI)
    stats["frac_invalid_range"] = float(bad_range.mean())
    x[bad_range] = np.nan

    # split valid samples into runs at jumps > JUMP_BPM
    valid = np.isfinite(x)
    run_id = np.full(n, -1)
    rid = -1
    prev = None
    for i in range(n):
        if not valid[i]:
            prev = None
            continue
        if prev is None or abs(x[i] - x[prev]) > JUMP_BPM:
            rid += 1
        run_id[i] = rid
        prev = i
    removed_short = np.zeros(n, bool)
    removed_level = np.zeros(n, bool)
    if rid >= 0:
        pos = np.where(run_id >= 0)[0]
        rr = run_id[pos]
        first = np.r_[True, rr[1:] != rr[:-1]]
        last = np.r_[rr[1:] != rr[:-1], True]
        starts = pos[first]
        ends = pos[last] + 1
        lens = ends - starts
        meds = np.array([np.median(x[s:e]) for s, e in zip(starts, ends)])
        short = lens < MIN_RUN_S * FS
        for r in np.where(short)[0]:
            removed_short[starts[r]:ends[r]] = True
        # level check against surrounding accepted (non-short) runs
        ctx = int(CONTEXT_S * FS)
        keep = ~short
        for r in (np.where(keep)[0] if USE_LEVEL_RULE else []):
            lo, hi = starts[r] - ctx, ends[r] + ctx
            nb = np.where(keep & (ends > lo) & (starts < hi) & (np.arange(len(starts)) != r))[0]
            if len(nb) == 0:
                continue
            w = lens[nb]
            # length-weighted median of neighbouring run medians
            order = np.argsort(meds[nb])
            cw = np.cumsum(w[order])
            ref = meds[nb][order][np.searchsorted(cw, cw[-1] / 2.0)]
            # only reject a run if it is not the dominant (longer) segment in its context
            if abs(meds[r] - ref) > JUMP_BPM and lens[r] < w.sum():
                removed_level[starts[r]:ends[r]] = True
    x[removed_short | removed_level] = np.nan
    stats["frac_removed_short_runs"] = float(removed_short.mean())
    stats["frac_removed_level_runs"] = float(removed_level.mean())

    y, m = interpolate_short_gaps(x, int(MAX_INTERP_S * FS))
    stats["frac_interpolated"] = float((m & ~np.isfinite(x)).mean())
    stats["frac_missing_final"] = float((~m).mean())
    return {"fhr": y, "mask": m, "stats": stats}


def clean_uc(uc: np.ndarray) -> dict:
    x = uc.astype(float).copy()
    stats = {}
    bad = ~np.isfinite(x) | (x < 0)
    zero = np.isfinite(x) & (x == 0)
    for s, e in _runs(zero):
        if (e - s) >= UC_ZERO_RUN_S * FS:
            bad[s:e] = True
    # flat-line detection (range within centred window)
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    w = int(UC_FLAT_WIN_S * FS)
    xf = np.where(bad, np.nan, x)
    filled = np.where(np.isfinite(xf), xf, 0.0)
    rng = maximum_filter1d(filled, w, mode="nearest") - minimum_filter1d(filled, w, mode="nearest")
    flat = (rng <= UC_FLAT_RANGE) & ~bad
    flat_final = np.zeros_like(flat)
    for s_, e_ in _runs(flat):
        if (e_ - s_) >= w:
            flat_final[s_:e_] = True
    stats["frac_flat"] = float(flat_final.mean())
    bad |= flat_final
    x[bad] = np.nan
    stats["frac_invalid"] = float(bad.mean())
    y, m = interpolate_short_gaps(x, int(MAX_INTERP_S * FS))
    stats["frac_missing_final"] = float((~m).mean())
    return {"uc": y, "mask": m, "stats": stats}
