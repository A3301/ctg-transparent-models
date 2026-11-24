"""Download the public expert-annotation sets used in this paper (no CTG outcome data).

Sources (approved by the authors on 2026-11-23; see docs/ek_veri_setleri.md):
  1. Romagnoli et al. 2020, Data in Brief, DOI 10.1016/j.dib.2020.105690 (PMC7256311).
     Supplement mmc2.zip, fetched via the Europe PMC supplementary-files endpoint.
     Licence of the article/supplement: CC BY-NC-ND 4.0 -> do NOT redistribute the files
     or modified versions; this script is what the reproducibility package ships instead.
  2. Hruban, Spilka et al. 2015, J Eval Clin Pract, DOI 10.1111/jep.12368.
     ExpertAnnCTU-UHB-CTG_20150203.csv, "free to use for non-commercial purposes" with citation.
  3. FHRMA (Boudet et al.), GitHub data-coeur/FHRMA (formerly utsb-fmm/FHRMA), GPL-3.0.
     Only the 66 training recordings and the expert analyses file are fetched, pinned to a commit.

Downloads are raw bytes saved to data/raw/annotations/<set>/ (git-ignored). Nothing is
extracted or executed here. SHA-256 of every file goes to MANIFEST.json next to the data and
to results/annotations/download_manifest.json (tracked); a re-run verifies against the latter.

Usage: .venv\\Scripts\\python -I src/data/fetch_annotations.py [--verify-only]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "annotations"
TRACKED = ROOT / "results" / "annotations" / "download_manifest.json"

# MD5 of mmc2.zip as listed in the PMC article XML (suppdata-md5)
ROMAGNOLI_MMC2_MD5 = "8566a2b5593292573957db0857be27b5"
FHRMA_COMMIT ="407a279ea55458b0512fd709646c5646dc2ef669"
FHRMA_RAW = f"https://raw.githubusercontent.com/data-coeur/FHRMA/{FHRMA_COMMIT}"

SETS = {
    "romagnoli2020": {
        "licence": "CC BY-NC-ND 4.0 (article licence; no separate data licence stated)",
        "citation": "Romagnoli S. et al., Data in Brief 2020, DOI 10.1016/j.dib.2020.105690",
        "files": {
            "europepmc_PMC7256311_supplementary.zip":
                "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7256311/supplementaryFiles",
        },
    },
    "hruban2015": {
        "licence": "free for non-commercial use, citing Hruban et al. 2015 (author page)",
        "citation": "Hruban L., Spilka J. et al., J Eval Clin Pract 2015;21(4):694-702, DOI 10.1111/jep.12368",
        "files": {
            "ExpertAnnCTU-UHB-CTG_20150203.csv":
                "https://people.ciirc.cvut.cz/~spilkjir/files/ExpertAnnCTU-UHB-CTG_20150203.csv",
        },
    },
    "fhrma": {
        "licence": "GPL-3.0 (repository licence)",
        "citation": "Boudet S. et al., FHRMA toolbox, SoftwareX 2020, DOI 10.1016/j.softx.2020.100428",
        "files": {
            "LICENSE": f"{FHRMA_RAW}/LICENSE",
            "expertAnalyses.mat": f"{FHRMA_RAW}/FHRMAdataset/analyses/expertAnalyses.mat",
            **{f"traindata/train{i:02d}.fhr": f"{FHRMA_RAW}/FHRMAdataset/traindata/train{i:02d}.fhr"
               for i in range(1, 67)},
        },
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "research-download-script"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - report and retry
            print(f"  attempt {attempt} failed for {dest.name}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed: {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    expected = {}
    if TRACKED.exists():
        for s, info in json.loads(TRACKED.read_text(encoding="utf-8"))["sets"].items():
            for name, meta in info["files"].items():
                expected[(s, name)] = meta["sha256"]

    manifest = {"downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "fhrma_commit": FHRMA_COMMIT, "sets": {}}
    bad = 0
    for s, info in SETS.items():
        files = {}
        for name, url in info["files"].items():
            dest = OUT / s / name
            if not args.verify_only and not dest.exists():
                print(f"[{s}] {name}")
                fetch(url, dest)
            if not dest.exists():
                print(f"MISSING {s}/{name}"); bad += 1; continue
            digest = sha256(dest)
            exp = expected.get((s, name))
            if s == "romagnoli2020":
                # Europe PMC builds this zip on the fly, so its own hash may change between
                # downloads; the publisher's MD5 of the data file inside it is the fixed check.
                import zipfile
                inner = hashlib.md5(zipfile.ZipFile(dest).read("mmc2.zip")).hexdigest()
                if inner != ROMAGNOLI_MMC2_MD5:
                    print(f"MMC2 MD5 MISMATCH {inner}"); bad += 1
                elif exp and exp != digest:
                    print("note: outer Europe PMC zip differs from the recorded one; mmc2.zip MD5 OK")
            elif exp and exp != digest:
                print(f"HASH MISMATCH {s}/{name}"); bad += 1
            files[name] = {"url": url, "bytes": dest.stat().st_size, "sha256": digest}
        manifest["sets"][s] = {"licence": info["licence"], "citation": info["citation"], "files": files}

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if not TRACKED.exists():
        TRACKED.parent.mkdir(parents=True, exist_ok=True)
        TRACKED.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n = sum(len(v["files"]) for v in manifest["sets"].values())
    tot = sum(f["bytes"] for v in manifest["sets"].values() for f in v["files"].values())
    print(f"{n} files, {tot / 1e6:.1f} MB, problems: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
