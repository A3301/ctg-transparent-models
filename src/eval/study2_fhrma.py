"""External concept check on FHRMA (docs/analiz_plani_v2.md §6.5), written before results were seen.

NeSy-v2 frame heads (repeat 0: 10 outer-fold ensembles x 5 members = 50 members, saved weights) are
applied to the 66 FHRMA training recordings, cut into consecutive 60-min windows (last one padded with
mask 0). v1 preprocessing (clean_fhr / clean_uc at 4 Hz, 1 Hz down-sampling, fixed scaling). TOCO
that is entirely zero -> UC mask 0. Expert 'notToAnalyse' and 'unreliableSignal' parts are excluded.
ACC / DEC event F1 of NeSy-v2 (median event thresholds over the 50 outer folds) vs the rule-based
extractor run on the whole record (as in docs/kavram_dogrulugu.md).

Run: .venv\\Scripts\\python -m src.eval.study2_fhrma
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import annotations as A  # noqa: E402
from src.data.inputs import FHR_C, FHR_S, UC_C, UC_S, _down  # noqa: E402
from src.eval.concept_fidelity import _intervals, match  # noqa: E402
from src.eval.events import BIN, MIN_BINS, bins_to_events, in_region  # noqa: E402
from src.features import figo  # noqa: E402
from src.models.nesy_v2 import NeSyV2  # noqa: E402
from src.preprocess.clean import clean_fhr, clean_uc  # noqa: E402

FS, WIN = 4, 3600 * 4


def load_members():
    nets = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "cv" / "nesy2_W60" / "weights" / "r0_f*_m*.pt"))):
        ck = torch.load(fp, map_location="cuda", weights_only=False)
        kw = {k: v for k, v in ck["cfg"].items() if k in ("width", "n_stages", "kernel", "drop")}
        net = NeSyV2(4, **kw).cuda()
        net.load_state_dict(ck["state"])
        net.eval()
        nets.append(net)
    assert len(nets) == 50, f"expected 50 members, found {len(nets)}"
    return nets


def thresholds():
    thr = [np.load(fp, allow_pickle=True)["event_thr"] for fp in
           sorted(glob.glob(str(ROOT / "results" / "cv" / "nesy2_W60" / "r*_f*.npz")))]
    return np.median(np.array(thr), 0)


def frame_probs(nets, fhr4, fm4, uc4, um4):
    n = len(fhr4)
    nw = int(np.ceil(n / WIN))
    pad = nw * WIN - n
    f, fm, u, um = (np.concatenate([a, np.zeros(pad, np.float32)]) for a in (fhr4, fm4, uc4, um4))
    f, fm = _down(f.reshape(nw, WIN), fm.reshape(nw, WIN), 4)
    u, um = _down(u.reshape(nw, WIN), um.reshape(nw, WIN), 4)
    f = np.where(fm > 0, (f - FHR_C) / FHR_S, 0.0)
    u = np.where(um > 0, (u - UC_C) / UC_S, 0.0)
    X = torch.tensor(np.stack([f, fm, u, um], 1), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        P = np.mean([net(X)["f_prob"].cpu().numpy() for net in nets], 0)     # (nw, 3, 900)
    return np.concatenate(list(P), axis=-1)[:, : int(np.ceil(n / BIN))]      # (3, n_bins)


def main():
    nets, thr = load_members(), thresholds()
    E = {k: v for k, v in A.fhrma_expert().items() if k.startswith("train")}
    counts = {"nesy2": {0: [], 1: []}, "rule": {0: [], 1: []}}
    for name in sorted(E):
        v = E[name]
        sig = A.fhrma_signal(name)
        n = len(sig["fhr1"])
        cf = clean_fhr(sig["fhr1"])
        fhr, fm = np.nan_to_num(cf["fhr"], nan=0.0).astype(np.float32), cf["mask"].astype(np.float32)
        if np.all(sig["toco"] == 0):
            uc, um = np.zeros(n, np.float32), np.zeros(n, np.float32)
        else:
            cu = clean_uc(sig["toco"])
            uc, um = np.nan_to_num(cu["uc"], nan=0.0).astype(np.float32), cu["mask"].astype(np.float32)
        P = frame_probs(nets, fhr, fm, uc, um)
        nb = P.shape[1]
        excl = np.zeros(nb * BIN, bool)
        for s, e in _intervals(v["notToAnalyse"], n) + _intervals(v["unreliableSignal"], n):
            excl[s:e] = True
        region = (~excl[: nb * BIN].reshape(nb, BIN)).mean(1) >= 0.5
        region &= (np.arange(nb) * BIN) < n
        valid = fm > 0.5
        base = figo.baseline_curve(fhr, valid)
        rule = {0: figo.detect_events(fhr, valid, base, +1), 1: figo.detect_events(fhr, valid, base, -1)}
        for lab, field in ((0, "accelerations"), (1, "decelerations")):
            ref = in_region([(s, e) for s, e in _intervals(v[field], n) if e > s], region.astype(float))
            for kind, det in (("nesy2", bins_to_events(P[lab], thr[lab], MIN_BINS[lab])),
                              ("rule", [(d["start"], d["end"]) for d in rule[lab]])):
                det = in_region(det, region.astype(float))
                counts[kind][lab].append([len(match(ref, det)), len(ref), len(det)])
    rng = np.random.default_rng(20261123)
    out = {"n_records": len(E), "thresholds": thr.tolist()}
    for lab, nm in ((0, "ACC"), (1, "DEC")):
        a, b = np.array(counts["nesy2"][lab]), np.array(counts["rule"][lab])
        f1 = lambda c: 2 * c[:, 0].sum() / (c[:, 1].sum() + c[:, 2].sum())  # noqa: E731
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(a), len(a))
            bs.append(f1(a[idx]) - f1(b[idx]))
        out[nm] = {"nesy2": {"se": a[:, 0].sum() / a[:, 1].sum(), "ppv": a[:, 0].sum() / max(a[:, 2].sum(), 1), "f1": f1(a)},
                   "rule": {"se": b[:, 0].sum() / b[:, 1].sum(), "ppv": b[:, 0].sum() / max(b[:, 2].sum(), 1), "f1": f1(b)},
                   "diff_f1": f1(a) - f1(b), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    p = ROOT / "results" / "study2" / "study2_fhrma.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
