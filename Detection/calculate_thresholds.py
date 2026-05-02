#!/usr/bin/env python3
import os
import re
import glob
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
#                 EDIT THESE PATHS ONLY
# ============================================================
BENIGN_DIR   = "path logs pf processing times"
OUT_DIR      = "Output path"
TXT_PATTERN  = "*.txt"

# Thresholds are computed from THIS ONE file:
# - "first_sorted": first file after sorting names (recommended: name it all1.txt)
# - OR set BASELINE_FILE to an exact path.
BASELINE_MODE = "first_sorted"   # "first_sorted" or "explicit"
BASELINE_FILE = ""               # if BASELINE_MODE="explicit", set full path here
# ============================================================


LINE_RE = re.compile(
    r"""
    ^(?P<fname>.+?\.jpg):\s+
    Original\s+(?P<orig_w>\d+)x(?P<orig_h>\d+),\s+
    Model\s+Input\s+(?P<in_w>\d+)x(?P<in_h>\d+),\s+
    Preprocess\s+(?P<tpre>[\d.]+)ms,\s+
    Inference\s+(?P<tinf>[\d.]+)ms,\s+
    Postprocess\(NMS\)\s+(?P<tpost>[\d.]+)ms,\s+
    Tracking\(SORT\)\s+(?P<ttrack>[\d.]+)ms,\s+
    Total\s+(?P<ttxt>[\d.]+)s,\s+
    Before\s+NMS\s+\(anchors\):\s+(?P<anchors>\d+),\s+
    Above\s+Conf\s+Thresh:\s+(?P<above>\d+),\s+
    After\s+NMS:\s+(?P<after>\d+),\s+
    Tracks:\s+(?P<tracks>\d+)
    """,
    re.VERBOSE
)


def parse_times_from_txt(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue

            tpre   = float(m.group("tpre"))
            tinf   = float(m.group("tinf"))
            tpost  = float(m.group("tpost"))
            ttrack = float(m.group("ttrack"))

            # compute total ourselves to stay consistent
            ttotal = tpre + tinf + tpost + ttrack

            rows.append({
                "tpre_ms": tpre,
                "tinf_ms": tinf,
                "tpost_ms": tpost,
                "ttrack_ms": ttrack,
                "ttotal_ms": ttotal,
            })

    return pd.DataFrame(rows)


def compute_thresholds(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(float)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=0))
    return {
        "max": float(np.max(x)),
        "mu": mu,
        "sigma": sigma,
        "mu+2sigma": mu + 2.0 * sigma,
        "mu+3sigma": mu + 3.0 * sigma,
        "p99": float(np.percentile(x, 99)),
        "p98": float(np.percentile(x, 98)),
        "p97": float(np.percentile(x, 97)),
        "p96": float(np.percentile(x, 96)),
        "p95": float(np.percentile(x, 95)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(BENIGN_DIR, "**", TXT_PATTERN), recursive=True))
    if not txts:
        raise RuntimeError(f"No txt files found in: {BENIGN_DIR}")

    if BASELINE_MODE == "explicit":
        if not BASELINE_FILE or not os.path.isfile(BASELINE_FILE):
            raise RuntimeError("BASELINE_MODE='explicit' but BASELINE_FILE is missing/invalid.")
        baseline_path = BASELINE_FILE
    else:
        baseline_path = txts[0]

    print("Baseline file for thresholds:", baseline_path)

    df = parse_times_from_txt(baseline_path)
    if df.empty:
        raise RuntimeError(f"Parsed 0 rows from baseline file (regex mismatch?): {baseline_path}")

    # Added preprocess + inference
    metrics = ["tpre_ms", "tinf_ms", "tpost_ms", "ttrack_ms", "ttotal_ms"]
    thr_types_order = ["max", "mu", "mu+2sigma", "mu+3sigma", "p99", "p98", "p97", "p96", "p95"]

    rows = []
    for m in metrics:
        thr = compute_thresholds(df[m].to_numpy())
        for t in thr_types_order:
            rows.append({
                "metric": m,
                "thr_type": t,
                "value": thr[t],
                "base_file": os.path.basename(baseline_path),
                "n_samples": int(len(df)),
                "mu": thr["mu"],
                "sigma": thr["sigma"],
            })

    out_csv = os.path.join(OUT_DIR, "thresholds_from_baseline_with_percentiles.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()