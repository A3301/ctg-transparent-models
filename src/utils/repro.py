"""Seeding and experiment logging helpers."""
from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "experiments" / "log.csv"
FIELDS = ["timestamp_utc", "phase", "experiment", "config", "seed", "metrics", "notes"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_experiment(phase: str, experiment: str, config: dict, seed: int,
                   metrics: dict, notes: str = "") -> None:
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phase": phase, "experiment": experiment,
            "config": json.dumps(config, sort_keys=True), "seed": seed,
            "metrics": json.dumps(metrics, sort_keys=True), "notes": notes,
        })
