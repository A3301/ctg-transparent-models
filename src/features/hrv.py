"""HRV-type FHR features used by Georgoulas et al. 2017 (VLF, LF power; Poincare SD2).

Approximation (documented): FHR at 4 Hz, gaps linearly interpolated over the whole window for
spectral estimation; Welch PSD (Hann, 1024-sample segments = 256 s); bands VLF 0-0.03 Hz,
LF 0.03-0.15 Hz. SD2 from the Poincare plot of successive 1-Hz FHR values (bpm), valid pairs only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def hrv_features(fhr, mask, fs=4):
    v = mask > 0.5
    if v.sum() < fs * 60 * 5:
        return dict(vlf=np.nan, lf=np.nan, sd2=np.nan)
    idx = np.arange(len(fhr))
    x = np.interp(idx, idx[v], fhr[v])
    x = x - x.mean()
    f, p = welch(x, fs=fs, nperseg=1024)
    df = f[1] - f[0]
    vlf = p[(f > 0) & (f <= 0.03)].sum() * df
    lf = p[(f > 0.03) & (f <= 0.15)].sum() * df
    n = len(fhr) // fs
    x1 = fhr[:n * fs].reshape(n, fs)
    m1 = v[:n * fs].reshape(n, fs)
    s = m1.sum(1)
    hz1 = np.where(s > 0, (x1 * m1).sum(1) / np.maximum(s, 1), np.nan)
    a, b = hz1[:-1], hz1[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    sd2 = float(np.std((a[ok] + b[ok]) / np.sqrt(2))) if ok.sum() > 10 else np.nan
    return dict(vlf=float(np.log(vlf + 1e-9)), lf=float(np.log(lf + 1e-9)), sd2=sd2)
