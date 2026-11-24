"""Loaders for the public expert-annotation sets (fetched by src/data/fetch_annotations.py).

These are reference labels for concept fidelity / expert comparison only. They are derived
from the CTG signal by clinicians, never from outcomes, but they are still NOT model inputs at
test time: expert event labels may only supervise concept heads inside training folds.

Romagnoli 2020 (CTU-UHB, 552 records)
    Each annotation_<rec>.mat holds `ann`, a 5 x L cell array aligned sample-by-sample with
    the 4 Hz CTU-UHB signal, and `dataloss` (1 x 3). Non-empty cells contain MATLAB `string`
    objects such as "(UC3" (onset of contraction 3) and ")UC3" (offset). scipy cannot decode
    MATLAB string objects, so `_mcos_strings` reads them from the __function_workspace__
    subsystem blob: each string object is saved as a uint64 array
    [version=1, ndims=2, 1, 1, nchar, utf16 chars packed little-endian]. The k-th decoded
    string corresponds to MCOS object id k (checked by `romagnoli_events` consistency tests).
Hruban/Spilka 2015
    CSV rec_id, expert_id, eval_step1..4 (9 experts; repeated rows = re-evaluations).
FHRMA (Boudet et al.)
    .fhr: uint32 unix timestamp, then 6-byte samples at 4 Hz: uint16 FHR1*4, uint16 FHR2*4,
    uint8 (unused here), uint8 TOCO*2 (layout from fhropen.m in the FHRMA repository).
    expertAnalyses.mat: struct array `data` with baseline/accelerations/decelerations etc.
"""
from __future__ import annotations

import io
import re
import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "raw" / "annotations"
ROM_ZIP = ANN / "romagnoli2020" / "europepmc_PMC7256311_supplementary.zip"
ROM_ROWS = 5
# e.g. "(UC3", ")UC3", "(DEC1E" (deceleration type suffix E/L/V/P)
_TOKEN = re.compile(r"^([()])([A-Za-z_]+?)(\d+)([A-Za-z]*)$")


# --------------------------------------------------------------------------- Romagnoli
def _mcos_strings(ws: bytes) -> list[list[str]]:
    """Decode MATLAB string objects (in saved order) from a __function_workspace__ blob.

    Each object may be a string array (e.g. 1x2 when an offset and the next onset fall on the
    same sample), so every object decodes to a list of strings.
    """
    out, i = [], 0
    while True:
        # miMATRIX(14) whose array-flags class is mxUINT64 (15)
        i = ws.find(b"\x0e\x00\x00\x00", i)
        if i < 0 or i + 48 > len(ws):
            break
        flags_tag = ws[i + 8:i + 16]
        if flags_tag != b"\x06\x00\x00\x00\x08\x00\x00\x00" or ws[i + 16] != 15:
            i += 4
            continue
        j = i + 24
        # dims (miINT32=5, 8 bytes) then name (small or normal element)
        dt, dn = struct.unpack_from("<II", ws, j)
        if dt != 5:
            i += 4
            continue
        j += 8 + dn + (-dn % 8)
        small = struct.unpack_from("<I", ws, j)[0]
        if small >> 16:  # small data element (name packed in 4 bytes)
            j += 8
        else:
            nl = struct.unpack_from("<I", ws, j + 4)[0]
            j += 8 + nl + (-nl % 8)
        t, n = struct.unpack_from("<II", ws, j)
        if t != 13:  # miUINT64
            i += 4
            continue
        payload = np.frombuffer(ws, dtype="<u8", count=n // 8, offset=j + 8)
        if payload.size >= 4 and payload[0] == 1 and 2 <= payload[1] <= 8:
            nd = int(payload[1])
            cnt = int(np.prod(payload[2:2 + nd]))
            lens = payload[2 + nd:2 + nd + cnt].astype(int)
            chars = payload[2 + nd + cnt:].tobytes()
            strs, p = [], 0
            for ln in lens:
                strs.append(chars[2 * p:2 * (p + ln)].decode("utf-16-le"))
                p += ln
            out.append(strs)
        i = j + 8 + n
    return out


@lru_cache(maxsize=1)
def _romagnoli_zip() -> zipfile.ZipFile:
    outer = zipfile.ZipFile(ROM_ZIP)
    return zipfile.ZipFile(io.BytesIO(outer.read("mmc2.zip")))


def romagnoli_record(rec: int | str) -> tuple[list[tuple[int, int, str]], np.ndarray, int]:
    """Return ([(row, sample, token)], dataloss, L) for one CTU-UHB record id."""
    m = sio.loadmat(io.BytesIO(_romagnoli_zip().read(f"annotation_{rec}.mat")), squeeze_me=False)
    ann = m["ann"]
    strings = _mcos_strings(m["__function_workspace__"].tobytes()) if "__function_workspace__" in m else []
    cells = []
    for c in range(ann.shape[1]):
        for r in range(ann.shape[0]):
            x = ann[r, c]
            if x.size == 0:
                continue
            # MatlabOpaque record: (s0='MCOS', s1='string', s2=uint32 ref); ref[4] = object id
            ref = np.asarray(x.ravel()[0][2]).ravel()
            cells.append((r, c, int(ref[4])))
    events = []
    for r, c, oid in cells:
        if not 1 <= oid <= len(strings):
            raise ValueError(f"record {rec}: object id {oid} out of range ({len(strings)} strings)")
        # within one cell, offsets are applied before onsets so ")UC3","(UC4" pairs correctly
        for tok in sorted(strings[oid - 1], key=lambda s: s[:1] != ")"):
            events.append((r, c, tok))
    return events, np.asarray(m["dataloss"], dtype=float).ravel(), ann.shape[1]


def romagnoli_events(rec: int | str) -> pd.DataFrame:
    """Pair onset/offset tokens into events (samples, 4 Hz).

    Columns: rec, row, label, idx, subtype, start, end, censored. Source quirks are kept, not
    silently dropped: an onset never closed (e.g. terminal bradycardia) gets end = annotation
    length and censored=True; an offset with no preceding onset and a repeated onset are
    listed in df.attrs["anomalies"] and not turned into events.
    """
    events, _, L = romagnoli_record(rec)
    opened, rows, anomalies = {}, [], []
    for r, c, tok in events:
        mt = _TOKEN.match(tok)
        if not mt:
            raise ValueError(f"record {rec}: unexpected token {tok!r}")
        side, lab, k, sub = mt.groups()
        key = (r, lab, int(k))
        if side == "(":
            if key in opened:
                anomalies.append(("duplicate_onset", tok, c))
            opened[key] = (c, sub)
        else:
            if key not in opened:
                anomalies.append(("offset_without_onset", tok, c))
                continue
            c0, sub0 = opened.pop(key)
            if sub and sub0 and sub != sub0:
                anomalies.append(("subtype_mismatch", tok, c))
            rows.append({"rec": int(rec), "row": r, "label": lab, "idx": int(k),
                         "subtype": sub0 or sub, "start": c0, "end": c, "censored": False})
    for (r, lab, k), (c0, sub0) in sorted(opened.items(), key=lambda kv: kv[1][0]):
        rows.append({"rec": int(rec), "row": r, "label": lab, "idx": k,
                     "subtype": sub0, "start": c0, "end": L, "censored": True})
    df = pd.DataFrame(rows, columns=["rec", "row", "label", "idx", "subtype", "start", "end", "censored"])
    df = df.sort_values(["start", "row"]).reset_index(drop=True)
    df.attrs["ann_len"] = L
    df.attrs["anomalies"] = anomalies
    return df


def romagnoli_records() -> list[int]:
    return sorted(int(n[len("annotation_"):-4]) for n in _romagnoli_zip().namelist())


# --------------------------------------------------------------------------- Hruban/Spilka
def hruban_expert_table() -> pd.DataFrame:
    return pd.read_csv(ANN / "hruban2015" / "ExpertAnnCTU-UHB-CTG_20150203.csv")


# --------------------------------------------------------------------------- FHRMA
def fhrma_signal(name: str) -> dict:
    """Read an FHRMA .fhr file -> dict(fhr1, fhr2, toco, timestamp); 4 Hz, bpm."""
    raw = (ANN / "fhrma" / "traindata" / name).read_bytes()
    ts = struct.unpack_from("<I", raw, 0)[0]
    body = raw[4:]
    n = len(body) // 6
    rec = np.frombuffer(body[:6 * n], dtype=np.dtype([("f1", "<u2"), ("f2", "<u2"), ("x", "u1"), ("toco", "u1")]))
    return {"fhr1": rec["f1"] / 4.0, "fhr2": rec["f2"] / 4.0, "toco": rec["toco"] / 2.0, "timestamp": ts}


def fhrma_expert() -> dict:
    """filename -> expert analysis struct fields (training records only carry annotations)."""
    m = sio.loadmat(ANN / "fhrma" / "expertAnalyses.mat", squeeze_me=True, struct_as_record=False)
    return {d.filename: {f: getattr(d, f) for f in d._fieldnames} for d in m["data"]}


# --------------------------------------------------------------------------- build
def build_romagnoli_table() -> None:
    """Write data/processed/romagnoli_events.csv (git-ignored: CC BY-NC-ND derivative) and
    results/annotations/romagnoli_summary.json (counts only)."""
    import json
    frames, meta, anomalies = [], [], []
    for r in romagnoli_records():
        df = romagnoli_events(r)
        _, dl, L = romagnoli_record(r)
        frames.append(df)
        anomalies += [{"rec": r, "kind": k, "token": t, "sample": int(c)} for k, t, c in df.attrs["anomalies"]]
        meta.append({"rec": r, "ann_len": int(L), "n_events": len(df), "dataloss": dl.round(3).tolist()})
    ev = pd.concat(frames, ignore_index=True)
    out = ROOT / "data" / "processed" / "romagnoli_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out, index=False)
    summ = {
        "records": len(meta),
        "records_without_events": sum(m["n_events"] == 0 for m in meta),
        "events_by_label": ev.groupby("label").size().to_dict(),
        "dec_subtypes": ev[ev.label == "DEC"].subtype.value_counts().to_dict(),
        "censored_events": int(ev.censored.sum()),
        "anomalies": anomalies,
        "ann_len_by_record": {m["rec"]: m["ann_len"] for m in meta},
    }
    js = ROOT / "results" / "annotations" / "romagnoli_summary.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(summ, indent=1, default=int), encoding="utf-8")
    print(f"{len(ev)} events -> {out}; summary -> {js}")


if __name__ == "__main__":
    build_romagnoli_table()
