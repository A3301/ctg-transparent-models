"""Evaluate predictor-block contrasts using existing held-out probabilities."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys
from zipfile import ZipFile

import numpy as np


EXPERIMENTS = {
    "hrv": "hrv_lr_W60",
    "hrv_figo": "transp_lr_noq_W60",
    "hrv_figo_quality": "transp_lr_W60",
}
CONTRASTS = (
    ("hrv_figo", "hrv"),
    ("hrv_figo_quality", "hrv_figo"),
    ("hrv_figo_quality", "hrv"),
)


class NumpyIdentifierUnpickler(pickle.Unpickler):
    """Allow only the ndarray reconstruction used by legacy string arrays."""

    def find_class(self, module, name):
        if (module, name) in {("numpy", "dtype"), ("numpy", "ndarray")}:
            return getattr(np, name)
        if name == "_reconstruct" and module in {"numpy._core.multiarray", "numpy.core.multiarray"}:
            return np._core.multiarray._reconstruct
        raise pickle.UnpicklingError(f"Unsupported object in legacy ndarray: {module}.{name}")


def read_identifiers(data, filename, key):
    try:
        values = data[key]
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
        with ZipFile(filename) as archive, archive.open(f"{key}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise ValueError(f"Unexpected NPY format in {filename}")
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            values = NumpyIdentifierUnpickler(stream).load()
            if values.shape != shape or values.dtype != dtype:
                raise ValueError(f"Invalid identifier array in {filename}")
    if values.ndim != 1 or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Non-string identifiers in {filename}")
    return list(map(str, values))


def read_held_out(folder):
    """Return fold memberships and one mean prediction per birth."""
    folds = {}
    observations = defaultdict(list)
    for repeat in range(5):
        seen = set()
        for fold in range(10):
            filename = folder / f"r{repeat}_f{fold}.npz"
            with np.load(filename, allow_pickle=False) as data:
                records = read_identifiers(data, filename, "records_te")
                validation = set(read_identifiers(data, filename, "records_va"))
                labels = data["y_te"].astype(int)
                scores = data["p_te"].astype(float)
            if (len(records) != len(set(records)) or seen.intersection(records)
                    or validation.intersection(records) or not np.isfinite(scores).all()):
                raise ValueError(f"Invalid held-out predictions in {filename}")
            folds[repeat, fold] = (frozenset(zip(records, labels)), frozenset(validation))
            seen.update(records)
            for record, label, score in zip(records, labels, scores, strict=True):
                observations[record].append((label, score))
        if len(seen) != 536:
            raise ValueError(f"Incomplete cohort in repeat {repeat} of {folder}")

    records = sorted(observations)
    labels, scores = [], []
    for record in records:
        values = observations[record]
        if len(values) != 5 or len({label for label, _ in values}) != 1:
            raise ValueError(f"Inconsistent repeat data for {record} in {folder}")
        labels.append(values[0][0])
        scores.append(np.mean([score for _, score in values]))
    return folds, records, np.asarray(labels), np.asarray(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_project", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_project.resolve()))
    from src.eval.metrics import auroc, bootstrap_ci, paired_bootstrap_diff

    model_data = {
        key: read_held_out(args.source_project / "results" / "cv" / experiment)
        for key, experiment in EXPERIMENTS.items()
    }
    base_folds, base_records, y, _ = model_data["hrv"]
    if len(y) != 536 or y.sum() != 38:
        raise ValueError("The cohort differs from the frozen W60 comparison")
    for key, (folds, records, labels, _) in model_data.items():
        if folds != base_folds or records != base_records or not np.array_equal(labels, y):
            raise ValueError(f"Mismatched folds, records or labels for {key}")

    # Match the source analysis's percentile record bootstrap and seed.
    bootstrap = {"n": 2000, "seed": 20261123}
    scores = {key: data[3] for key, data in model_data.items()}
    results = {"cohort_records": len(y), "events": int(y.sum()), "folds": len(base_folds),
               "bootstrap": bootstrap, "source_experiments": EXPERIMENTS,
               "models": {}, "contrasts": {}, "status": "post hoc fixed-prediction reanalysis"}
    for key, values in scores.items():
        point, lower, upper = bootstrap_ci(y, values, auroc, **bootstrap)
        results["models"][key] = {"auroc": point, "ci95": [lower, upper]}
    for left, right in CONTRASTS:
        point, lower, upper, _ = paired_bootstrap_diff(y, scores[left], scores[right], auroc, **bootstrap)
        results["contrasts"][f"{left}_minus_{right}"] = {"delta_auroc": point, "ci95": [lower, upper]}

    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
