"""HRV features (Georgoulas 2017) per window; S1_30 = last 30 min of stage I (consistency check)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features.hrv import hrv_features  # noqa: E402
out = ROOT / "results" / "features"
for w, src, cut in [("W60", "W60", None), ("S1_60", "S1_60", None), ("W30", "W30", None), ("W90", "W90", None), ("S1_30", "S1_60", 7200)]:
    z = np.load(ROOT / "data" / "processed" / f"windows_{src}.npz")
    f, m = z["fhr"], z["fhr_mask"]
    if cut: f, m = f[:, -cut:], m[:, -cut:]
    rows = [hrv_features(f[i], m[i]) for i in range(len(f))]
    df = pd.DataFrame(rows, index=pd.Index(z["records"].astype(str), name="record"))
    df.to_csv(out / f"hrv_{w}.csv"); print(w, df.isna().sum().to_dict(), df.median().round(2).to_dict())
