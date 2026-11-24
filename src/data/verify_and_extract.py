"""Verify the copied CTU-UHB zip against the archive record, extract it, and
verify every member against the provider SHA256SUMS manifest."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ZIP = RAW / "ctu-uhb-ctgdb_1.0.0.zip"
MANIFEST = RAW / "SHA256SUMS_provider.txt"
EXPECTED_ZIP_SHA = "50619d820bd35856674c688ab77fb0ea46e61276c57e8795b2a1b81a45f07525"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    zsha = sha256(ZIP)
    assert zsha == EXPECTED_ZIP_SHA, f"zip hash mismatch: {zsha}"
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        z.extractall(RAW)
    top = sorted({n.split("/")[0] for n in names})
    assert len(top) == 1, top
    db = RAW / top[0]
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            expected[name.strip()] = h
    ok, bad, missing = 0, [], []
    for name, h in expected.items():
        p = db / name
        if not p.exists():
            missing.append(name)
        elif sha256(p) != h:
            bad.append(name)
        else:
            ok += 1
    report = {
        "zip_sha256": zsha,
        "zip_matches_archive_record": True,
        "db_dir": str(db.relative_to(ROOT)),
        "manifest_entries": len(expected),
        "verified_ok": ok,
        "mismatch": bad,
        "missing": missing,
        "zip_members": len(names),
    }
    (RAW / "verification_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
