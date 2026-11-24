"""Rule-based FIGO 2015 feature extractor (4 Hz, one analysis window).

Produces
  * numeric FIGO features for the FIGO-LR / FIGO-XGB baselines,
  * record-level weak concept labels for the neuro-symbolic model,
  * a rule-based FIGO 2015 class (normal / suspicious / pathological).

Every threshold is defined in THRESHOLDS below and documented in docs/faz_3_rapor.md.
References: Ayres-de-Campos et al., FIGO consensus guidelines on intrapartum fetal
monitoring: cardiotocography, Int J Gynecol Obstet 2015 (FIGO 2015).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FS = 4
THRESHOLDS = dict(
    # baseline: iterative robust estimate, +/-5 min window, outliers > 10 bpm excluded (2 passes)
    baseline_half_window_s=300, baseline_exclude_bpm=10.0,
    # accelerations / decelerations (FIGO: abrupt rise/fall > 15 bpm lasting > 15 s)
    acc_amp=15.0, acc_min_s=15.0, acc_max_s=600.0,
    dec_amp=15.0, dec_min_s=15.0,
    prolonged_s=180.0,            # FIGO prolonged deceleration > 3 min
    very_prolonged_s=300.0,       # FIGO pathological: deceleration > 5 min
    abrupt_onset_s=30.0,          # onset-to-nadir < 30 s -> variable (abrupt) morphology
    late_lag_s=20.0,              # nadir > 20 s after contraction peak -> late (gradual decels)
    early_lag_s=20.0,             # |nadir - peak| <= 20 s -> early (gradual decels)
    assoc_window_s=60.0,          # decel associated with a contraction if nadir within peak -60..+60 s
    # variability: per-minute bandwidth (P95-P5) of FHR after removing acc/dec
    var_reduced=5.0, var_increased=25.0,
    # contractions: prominent peaks (prominence >= 10 units) of 15-s smoothed UC, >= 60 s apart
    uc_smooth_s=15.0, uc_amp=10.0, uc_min_gap_s=60.0, uc_min_halfwidth_s=15.0, uc_max_halfwidth_s=180.0,
    # sinusoidal: 10-min segments, dominant period 3-5 cycles/min, amplitude 5-15 bpm
    sin_band_hz=(3 / 60.0, 5 / 60.0), sin_power_frac=0.5, sin_amp=(5.0, 15.0),
    # repetitive: decelerations with > 50 % of contractions (FIGO 2015)
    repetitive_frac=0.5,
    # FIGO duration criteria scaled to the window
    reduced_var_path_min=50.0, increased_var_path_min=30.0, sinus_path_min=30.0,
    rep_dec_path_min=30.0,
)
T = THRESHOLDS


def _runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def baseline_curve(fhr, valid):
    """Robust moving baseline (bpm) over the window; NaN where no data at all."""
    x = np.where(valid, fhr, np.nan)
    hw = int(T["baseline_half_window_s"] * FS)
    step = FS * 10  # evaluate every 10 s and interpolate
    centers = np.arange(0, len(x), step)
    est = np.full(len(centers), np.nan)
    for k, c in enumerate(centers):
        seg = x[max(0, c - hw):c + hw]
        seg = seg[np.isfinite(seg)]
        if seg.size < FS * 60:
            continue
        b = np.median(seg)
        for _ in range(2):
            keep = seg[np.abs(seg - b) <= T["baseline_exclude_bpm"]]
            if keep.size < FS * 30:
                break
            b = keep.mean()
        est[k] = b
    ok = np.isfinite(est)
    if not ok.any():
        return np.full(len(x), np.nan)
    return np.interp(np.arange(len(x)), centers[ok], est[ok])


def detect_events(fhr, valid, base, sign):
    """Accelerations (sign=+1) or decelerations (sign=-1). Returns list of dicts."""
    amp = T["acc_amp"] if sign > 0 else T["dec_amp"]
    d = sign * (fhr - base)
    above = valid & np.isfinite(base) & (d > amp)
    events = []
    for s, e in _runs(above):
        dur = (e - s) / FS
        if dur < (T["acc_min_s"] if sign > 0 else T["dec_min_s"]):
            continue
        if sign > 0 and dur > T["acc_max_s"]:
            continue
        # extend to where the curve leaves / returns to baseline (d > 0)
        s0, e0 = s, e
        while s0 > 0 and valid[s0 - 1] and d[s0 - 1] > 0:
            s0 -= 1
        while e0 < len(d) and valid[e0] and d[e0] > 0:
            e0 += 1
        seg = d[s0:e0]
        peak = s0 + int(np.argmax(seg))
        events.append(dict(start=s0, end=e0, peak=peak, dur_s=(e0 - s0) / FS,
                           depth=float(seg.max()), area=float(np.nansum(seg) / FS),
                           onset_to_peak_s=(peak - s0) / FS))
    # merge events whose extended bounds overlap
    merged = []
    for ev in sorted(events, key=lambda z: z["start"]):
        if merged and ev["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
        else:
            merged.append(dict(ev))
    # shape measures of a merged event are computed once over its merged bounds; summing the parts' areas
    # counted overlapping samples twice (deviation log #13)
    for m in merged:
        seg = d[m["start"]:m["end"]]
        peak = m["start"] + int(np.nanargmax(seg))
        m.update(peak=peak, depth=float(np.nanmax(seg)), area=float(np.nansum(seg) / FS),
                 dur_s=(m["end"] - m["start"]) / FS, onset_to_peak_s=(peak - m["start"]) / FS)
    return merged


def detect_contractions(uc, uc_valid):
    """Contractions as prominent peaks of the smoothed UC signal.
    (A rolling-median baseline was tried first and rejected: with frequent contractions the
    median sits at contraction level and most contractions were missed; see faz_3_rapor.)"""
    from scipy.signal import find_peaks
    x = np.where(uc_valid, uc, np.nan)
    if np.isfinite(x).mean() < 0.05:
        return [], np.zeros(len(uc), bool)
    filled = np.where(np.isfinite(x), x, np.nanmedian(x))
    sm = uniform_filter1d(filled, int(T["uc_smooth_s"] * FS))
    pk, prop = find_peaks(sm, prominence=T["uc_amp"], distance=int(T["uc_min_gap_s"] * FS),
                          width=(T["uc_min_halfwidth_s"] * FS, T["uc_max_halfwidth_s"] * FS), rel_height=0.5)
    cons = []
    for j, p_ in enumerate(pk):
        s_, e_ = int(prop["left_ips"][j]), int(prop["right_ips"][j]) + 1
        if uc_valid[max(0, s_):e_].mean() < 0.5:
            continue
        cons.append(dict(start=s_, end=e_, peak=int(p_), prominence=float(prop["prominences"][j])))
    return cons, uc_valid


def classify_decels(decs, cons, uc_valid):
    """Assign FIGO type to each deceleration; returns per-contraction association info."""
    aw = int(T["assoc_window_s"] * FS)
    for dv in decs:
        dv["type"] = "unknown"
        if dv["dur_s"] > T["prolonged_s"]:
            dv["type"] = "prolonged"
            continue
        abrupt = dv["onset_to_peak_s"] < T["abrupt_onset_s"]
        # nearest contraction peak to the nadir
        best, lag = None, None
        for c in cons:
            l = (dv["peak"] - c["peak"]) / FS
            if -aw / FS <= l <= aw / FS and (lag is None or abs(l) < abs(lag)):
                best, lag = c, l
        dv["assoc"] = best is not None
        dv["lag_s"] = lag
        uc_known = uc_valid[max(0, dv["start"] - aw):dv["end"]].mean() > 0.5
        if abrupt:
            dv["type"] = "variable"
        elif best is not None:
            if abs(lag) <= T["early_lag_s"]:
                dv["type"] = "early"
            elif lag > T["late_lag_s"]:
                dv["type"] = "late"
            else:
                dv["type"] = "unassociated"  # nadir well before the contraction peak
        elif not uc_known:
            dv["type"] = "unknown"
        else:
            dv["type"] = "unassociated"
    return decs


def variability_minutes(fhr, valid, base, accs, decs):
    """Per-minute bandwidth (P95-P5) after removing acc/dec samples; NaN if < 50 % data.
    max-min was tried first and rejected (31 % of records "increased", driven by spikes)."""
    keep = valid.copy()
    for ev in accs + decs:
        keep[ev["start"]:ev["end"]] = False
    n_min = len(fhr) // (60 * FS)
    amps = np.full(n_min, np.nan)
    for m in range(n_min):
        sl = slice(m * 60 * FS, (m + 1) * 60 * FS)
        seg = fhr[sl][keep[sl]]
        if seg.size >= 30 * FS:
            amps[m] = np.percentile(seg, 95) - np.percentile(seg, 5)  # robust bandwidth
    return amps


def stv_ltv(fhr, valid):
    """Dawes-Redman style STV/LTV on 3.75 s epochs (pulse interval, ms)."""
    ep = int(3.75 * FS)
    n = len(fhr) // ep
    pi = np.full(n, np.nan)
    for k in range(n):
        seg = fhr[k * ep:(k + 1) * ep][valid[k * ep:(k + 1) * ep]]
        if seg.size >= ep // 2:
            pi[k] = 60000.0 / seg.mean()
    d = np.abs(np.diff(pi))
    per_min = []
    for m in range(n // 16):
        dm = d[m * 16:(m + 1) * 16 - 1]
        if np.isfinite(dm).sum() >= 8:
            per_min.append(np.nanmean(dm))
    stv = float(np.mean(per_min)) if per_min else np.nan
    ltv = []
    for m in range(n // 16):
        p = pi[m * 16:(m + 1) * 16]
        if np.isfinite(p).sum() >= 8:
            ltv.append(np.nanmax(p) - np.nanmin(p))
    return stv, (float(np.mean(ltv)) if ltv else np.nan)


def sinusoidal_minutes(fhr, valid, base):
    """Minutes (in 10-min blocks) classified as sinusoidal."""
    seg_len = 600 * FS
    sin_min = 0.0
    for s in range(0, len(fhr) - seg_len + 1, seg_len):
        v = valid[s:s + seg_len]
        if v.mean() < 0.8:
            continue
        x = np.where(v, fhr[s:s + seg_len] - base[s:s + seg_len], 0.0)
        x = x - x.mean()
        f = np.fft.rfftfreq(len(x), 1 / FS)
        p = np.abs(np.fft.rfft(x)) ** 2
        band = (f >= T["sin_band_hz"][0]) & (f <= T["sin_band_hz"][1])
        tot = p[(f > 0.01) & (f < 0.5)].sum()
        amp = np.percentile(x, 95) - np.percentile(x, 5)
        if tot > 0 and p[band].sum() / tot > T["sin_power_frac"] and T["sin_amp"][0] <= amp <= 2 * T["sin_amp"][1]:
            sin_min += 10.0
    return sin_min


def events_from_intervals(fhr, valid, base, intervals, sign):
    """Event dicts (same fields as detect_events) for externally detected (start, end) intervals in 4 Hz
    samples, e.g. from the expert-supervised detector (Plan B, analiz_plani_v3 §4.2). Intervals with no
    valid sample relative to a defined baseline are dropped."""
    out = []
    for s, e in intervals:
        s, e = int(max(0, s)), int(min(len(fhr), e))
        if e <= s:
            continue
        ok = valid[s:e] & np.isfinite(base[s:e])
        if not ok.any():
            continue
        d = np.where(ok, sign * (fhr[s:e] - base[s:e]), np.nan)
        peak = s + int(np.nanargmax(d))
        out.append(dict(start=s, end=e, peak=peak, dur_s=(e - s) / FS, depth=float(np.nanmax(d)),
                        area=float(np.nansum(np.clip(d, 0, None)) / FS), onset_to_peak_s=(peak - s) / FS))
    return out


def contractions_from_intervals(uc, uc_valid, intervals):
    cons = []
    for s, e in intervals:
        s, e = int(max(0, s)), int(min(len(uc), e))
        if e <= s or uc_valid[s:e].mean() < 0.5:
            continue
        x = np.where(uc_valid[s:e], uc[s:e], -np.inf)
        cons.append(dict(start=s, end=e, peak=s + int(np.argmax(x)), prominence=np.nan))
    return cons


def extract(fhr, fhr_mask, uc, uc_mask, events=None) -> dict:
    """All features for one window. Inputs at 4 Hz; masks are 0/1.
    `events` (optional) = {"acc": [(s, e)], "dec": [(s, e)], "uc": [(s, e)]} replaces the rule-based
    ACC / DEC / contraction detection; everything else (baseline, typing rules, variability) is unchanged."""
    valid = fhr_mask > 0.5
    uc_valid = uc_mask > 0.5
    win_min = len(fhr) / FS / 60
    base = baseline_curve(fhr, valid)
    if events is None:
        accs = detect_events(fhr, valid, base, +1)
        decs = detect_events(fhr, valid, base, -1)
        cons, _ = detect_contractions(uc, uc_valid)
    else:
        accs = events_from_intervals(fhr, valid, base, events["acc"], +1)
        decs = events_from_intervals(fhr, valid, base, events["dec"], -1)
        cons = contractions_from_intervals(uc, uc_valid, events["uc"])
    decs = classify_decels(decs, cons, uc_valid)
    amps = variability_minutes(fhr, valid, base, accs, decs)
    stv, ltv = stv_ltv(fhr, valid)
    sin_min = sinusoidal_minutes(fhr, valid, base)

    bl = float(np.nanmedian(base[valid])) if valid.any() and np.isfinite(base).any() else np.nan
    var_med = float(np.nanmedian(amps)) if np.isfinite(amps).any() else np.nan
    n_amp = max(1, int(np.isfinite(amps).sum()))
    red_min = float(np.nansum(amps < T["var_reduced"]))
    inc_min = float(np.nansum(amps > T["var_increased"]))
    types = [d["type"] for d in decs]
    n_c = len(cons)
    # repetitive: fraction of contractions associated with a decel of a given type
    def assoc_frac(t):
        if n_c == 0:
            return np.nan
        cnt = 0
        for c in cons:
            if any((t is None or d.get("type") == t) and d.get("lag_s") is not None and
                   abs((d["peak"] - c["peak"]) / FS) <= T["assoc_window_s"] for d in decs):
                cnt += 1
        return cnt / n_c
    late_frac, var_frac, early_frac = assoc_frac("late"), assoc_frac("variable"), assoc_frac("early")
    any_frac = assoc_frac(None)
    rep_minutes = lambda t: float(sum(d["dur_s"] for d in decs if d["type"] == t) / 60)  # noqa: E731
    feats = dict(
        baseline=bl, stv_ms=stv, ltv_ms=ltv, var_amp_median=var_med,
        reduced_var_min=red_min * win_min / n_amp, increased_var_min=inc_min * win_min / n_amp,
        n_acc=len(accs), n_dec=len(decs),
        n_dec_early=types.count("early"), n_dec_late=types.count("late"),
        n_dec_variable=types.count("variable"), n_dec_prolonged=types.count("prolonged"),
        n_dec_other=types.count("unassociated") + types.count("unknown"),
        dec_total_min=float(sum(d["dur_s"] for d in decs) / 60),
        dec_max_s=float(max([d["dur_s"] for d in decs], default=0.0)),
        dec_area=float(sum(d["area"] for d in decs)),
        dec_max_depth=float(max([d["depth"] for d in decs], default=0.0)),
        late_frac_contr=late_frac, variable_frac_contr=var_frac, early_frac_contr=early_frac,
        any_dec_frac_contr=any_frac,
        sinusoidal_min=sin_min, uc_per_10min=n_c / win_min * 10 if uc_valid.mean() > 0.3 else np.nan,
        fhr_valid_frac=float(valid.mean()), uc_valid_frac=float(uc_valid.mean()),
    )
    feats.update(concepts_and_class(feats, decs, win_min))
    return feats


def concepts_and_class(f: dict, decs: list, win_min: float) -> dict:
    """Weak concept labels and rule-based FIGO 2015 class."""
    c = {}
    bl = f["baseline"]
    c["c_baseline"] = (np.nan if not np.isfinite(bl) else
                       0 if bl < 100 else 1 if bl < 110 else 2 if bl <= 160 else 3)  # sev.brady, brady, normal, tachy
    if f["sinusoidal_min"] >= T["sinus_path_min"] * win_min / 60:
        c["c_variability"] = 3                                            # sinusoidal
    elif not np.isfinite(f["var_amp_median"]):
        c["c_variability"] = np.nan
    else:
        c["c_variability"] = 0 if f["var_amp_median"] < T["var_reduced"] else (2 if f["var_amp_median"] > T["var_increased"] else 1)
    c["c_accel"] = int(f["n_acc"] > 0)
    rf = T["repetitive_frac"]
    c["c_rep_late"] = int(np.isfinite(f["late_frac_contr"]) and f["late_frac_contr"] > rf)
    c["c_rep_variable"] = int(np.isfinite(f["variable_frac_contr"]) and f["variable_frac_contr"] > rf)
    c["c_rep_any"] = int(np.isfinite(f["any_dec_frac_contr"]) and f["any_dec_frac_contr"] > rf)
    c["c_early"] = int(f["n_dec_early"] > 0)
    c["c_prolonged"] = int(f["n_dec_prolonged"] > 0)
    c["c_very_prolonged"] = int(f["dec_max_s"] > T["very_prolonged_s"])
    scale = win_min / 60.0
    c["c_reduced_var_long"] = int(f["reduced_var_min"] > T["reduced_var_path_min"] * scale)
    c["c_increased_var_long"] = int(f["increased_var_min"] > T["increased_var_path_min"] * scale)
    # repetitive late / prolonged decels lasting > 30 min (20 min if reduced variability)
    rep_lp_min = sum(d["dur_s"] for d in decs if d["type"] in ("late", "prolonged")) / 60
    rep_span = 0.0
    lp = [d for d in decs if d["type"] in ("late", "prolonged")]
    if len(lp) >= 2:
        rep_span = (lp[-1]["end"] - lp[0]["start"]) / FS / 60
    lim = (20.0 if c["c_reduced_var_long"] or c["c_variability"] == 0 else 30.0) * scale
    c["c_rep_late_long"] = int((c["c_rep_late"] or c["c_prolonged"]) and rep_span > lim)
    # FIGO 2015
    pathological = (c["c_baseline"] == 0) or c["c_reduced_var_long"] or c["c_increased_var_long"] \
        or c["c_variability"] == 3 or c["c_rep_late_long"] or c["c_very_prolonged"]
    normal = (c["c_baseline"] == 2) and (c["c_variability"] == 1) and not (c["c_rep_any"] or c["c_prolonged"])
    c["figo_class"] = 2 if pathological else (0 if normal else 1)
    c["rep_lp_minutes"] = rep_lp_min
    return c
