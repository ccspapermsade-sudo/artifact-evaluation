#!/usr/bin/env python3
import os
import re
import glob
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
#                 EDIT THESE ONLY
# ============================================================
#Log format:n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984252904844.jpg: Original 1600x900, Model Input 1920x1088, Preprocess 9.54ms, Inference 4.76ms, Postprocess(NMS) 1.30ms, Tracking(SORT) 56.20ms, Total 0.07s, Before NMS (anchors): 128520, Above Conf Thresh: 3256, After NMS: 210, Tracks: 198


LOG_DIR = "log path"

OUT_DIR = "Output path"

TXT_PATTERN = "*.txt"

FPS = 30
WINDOW_S = 1
WINDOW_STEP_S = 1   # keep equal to WINDOW_S for non-overlapping windows

# These thresholds are only saved as extra window statistics.
# Rules 1--5 below use late_frames L_k, not these thresholds.
TPOST_THR_MS = 1.46
TTRACK_THR_MS = 3.16
TTOTAL_THR_MS = 31.69

# Rule parameters
NUM_CAMERAS = 3
DANGER_EXTRA = 1     # tau_D = C + 1 = 7 when WINDOW_S = 1

RULE1_CONW = 3     # r1: sustained mild degradation
RULE2_CONW = 3     # r2: persistent overload
RULE4_ETA = 3      # eta: more than 3 Rule-3 alerts in 60 seconds triggers Rule 4

# Rule 5 threshold:
# If None, uses FPS * WINDOW_S - tau_D
RULE5_LATE_THR = None

# ============================================================


# -----------------------------
# Parsing
# -----------------------------
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


@dataclass
class RunData:
    run_name: str
    df: pd.DataFrame


def parse_txt_file(path: str) -> RunData:
    rows = []
    run_name = os.path.basename(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            m = LINE_RE.match(line)

            if not m:
                continue

            tpre = float(m.group("tpre"))
            tinf = float(m.group("tinf"))
            tpost = float(m.group("tpost"))
            ttrack = float(m.group("ttrack"))

            # Compute total ourselves in ms
            ttotal = tpre + tinf + tpost + ttrack

            rows.append({
                "fname": m.group("fname"),
                "tpre_ms": tpre,
                "tinf_ms": tinf,
                "tpost_ms": tpost,
                "ttrack_ms": ttrack,
                "ttotal_ms": ttotal,
                "above_conf": int(m.group("above")),
                "after_nms": int(m.group("after")),
                "tracks": int(m.group("tracks")),
            })

    df = pd.DataFrame(rows).reset_index(drop=True)
    return RunData(run_name=run_name, df=df)


# -----------------------------
# Late frames per window
# -----------------------------
def late_frames_window_budget(ttotal_ms_window: np.ndarray, window_s: int) -> int:
    """
    Compute L_k for one window.

    The window has a time budget of WINDOW_S * 1000 ms.
    Frames are processed sequentially until the budget is exceeded.
    Remaining frames are late/stale/dropped.
    """
    budget_ms = float(window_s) * 1000.0

    cum = 0.0
    processed = 0

    for tms in ttotal_ms_window:
        if cum + float(tms) <= budget_ms + 1e-9:
            cum += float(tms)
            processed += 1
        else:
            break

    late = len(ttotal_ms_window) - processed
    return max(0, int(late))


# -----------------------------
# Window evaluation
# -----------------------------
def eval_windows_detector_fps_budget(
    run: RunData,
    fps: float,
    window_s: int,
    step_s: int,
    tpost_thr: float,
    ttrack_thr: float,
    ttotal_thr: float
) -> pd.DataFrame:

    df = run.df

    if df.empty:
        return pd.DataFrame()

    n_total = len(df)

    frames_per_window = max(1, int(round(window_s * fps)))
    frames_per_step = max(1, int(round(step_s * fps)))

    tpost = df["tpost_ms"].to_numpy(dtype=np.float64)
    ttrack = df["ttrack_ms"].to_numpy(dtype=np.float64)
    ttotal = df["ttotal_ms"].to_numpy(dtype=np.float64)

    ex_post = (tpost > float(tpost_thr)).astype(np.int8)
    ex_track = (ttrack > float(ttrack_thr)).astype(np.int8)
    ex_total = (ttotal > float(ttotal_thr)).astype(np.int8)

    rows = []

    for i0 in range(0, n_total, frames_per_step):
        i1 = min(n_total, i0 + frames_per_window)

        if i1 <= i0:
            continue

        win_idx = i0 // frames_per_step

        Lk = late_frames_window_budget(
            ttotal_ms_window=ttotal[i0:i1],
            window_s=window_s
        )

        rows.append({
            "run": run.run_name,
            "window_idx": int(win_idx),
            "time_s": float(win_idx * step_s),
            "frame_start": int(i0),
            "frame_end": int(i1),
            "n_frames": int(i1 - i0),

            # This is L_k
            "late_frames": int(Lk),

            # Extra useful window-level statistics
            "window_total_latency_ms": float(np.sum(ttotal[i0:i1])),
            "mean_total_ms": float(np.mean(ttotal[i0:i1])),
            "p99_total_ms": float(np.percentile(ttotal[i0:i1], 99)),
            "max_total_ms": float(np.max(ttotal[i0:i1])),

            "mean_pre_ms": float(np.mean(df["tpre_ms"].iloc[i0:i1])),
            "mean_inf_ms": float(np.mean(df["tinf_ms"].iloc[i0:i1])),
            "mean_post_ms": float(np.mean(df["tpost_ms"].iloc[i0:i1])),
            "mean_track_ms": float(np.mean(df["ttrack_ms"].iloc[i0:i1])),

            # Counts exceeding the optional latency thresholds
            "ex_post_cnt": int(ex_post[i0:i1].sum()),
            "ex_track_cnt": int(ex_track[i0:i1].sum()),
            "ex_total_cnt": int(ex_total[i0:i1].sum()),
        })

    return pd.DataFrame(rows)


# -----------------------------
# Rule helpers
# -----------------------------
def _count_sliding_all_true(cond: np.ndarray, length: int) -> int:
    """
    Count sliding segments where all values are True.

    Example:
    cond = [T, T, T, F, T, T, T]
    length = 3
    count = 2
    """
    length = int(length)

    if length <= 0:
        return 0

    n = len(cond)

    if n < length:
        return 0

    x = cond.astype(np.int32)

    c = np.zeros(n + 1, dtype=np.int32)
    c[1:] = np.cumsum(x)

    win_sums = c[length:] - c[:-length]

    return int(np.sum(win_sums == length))


def _count_rule4_alerts_reset_after_trigger(
    rule3_flags: np.ndarray,
    window_step_s: int,
    eta: int
) -> int:
    """
    Rule 4:
    Generate one Rule-4 alert every time more than eta Rule-3 alerts
    occur within a 60-second interval.

    After generating one Rule-4 alert, reset the buffer so the same
    Rule-3 alerts are not reused.

    Example:
    eta = 3
    Rule-3 alerts at seconds:
    1, 5, 8, 12, 15, 28, 34, 50

    Output:
    Rule-4 alerts = 2
    Alert 1: 1, 5, 8, 12
    Alert 2: 15, 28, 34, 50
    """
    step = int(window_step_s)

    if step <= 0:
        step = 1

    max_window_len = int(round(60.0 / float(step)))
    needed_alerts = int(eta) + 1

    # indices/windows where Rule 3 fired
    alert_indices = np.where(rule3_flags.astype(np.int8) == 1)[0]

    rule4_alerts = 0
    buffer = []

    for idx in alert_indices:
        buffer.append(int(idx))

        # Remove old Rule-3 alerts outside the 60-second interval
        while buffer and (idx - buffer[0]) >= max_window_len:
            buffer.pop(0)

        # More than eta Rule-3 alerts within 60 seconds
        if len(buffer) >= needed_alerts:
            rule4_alerts += 1

            # Reset, so these Rule-3 alerts are not reused
            buffer = []

    return int(rule4_alerts)


def apply_rules_per_run(
    dfw_run: pd.DataFrame,
    fps: float,
    window_s: int,
    window_step_s: int,
    rule1_conw: int,
    rule2_conw: int,
    rule4_eta: int,
    num_cameras: int,
    danger_extra: int,
    rule5_late_thr: Optional[int]
) -> dict:

    dfw_run = dfw_run.sort_values("window_idx").reset_index(drop=True)

    L = dfw_run["late_frames"].to_numpy(dtype=np.int32)

    # tau_D = (C + 1) * WINDOW_S
    # For C = 6 and WINDOW_S = 1, tau_D = 7
    tau_D = max(1, (int(num_cameras) + int(danger_extra)) * int(window_s))

    frames_per_window = int(round(float(fps) * float(window_s)))

    if rule5_late_thr is None:
        rule5_thr = frames_per_window - int(tau_D)
    else:
        rule5_thr = int(rule5_late_thr)

    rule5_thr = max(0, rule5_thr)

    # ========================================================
    # Rule 1: sustained mild degradation
    # If for r1 consecutive windows:
    # 2 <= L_j < tau_D
    # ========================================================
    cond_r1 = (L >= 2) & (L < tau_D)
    rule1_alerts = _count_sliding_all_true(
        cond=cond_r1,
        length=int(rule1_conw)
    )

    # ========================================================
    # Rule 2: persistent overload
    # If for r2 consecutive windows:
    # L_j >= tau_D
    # ========================================================
    cond_r2 = (L >= tau_D)
    rule2_alerts = _count_sliding_all_true(
        cond=cond_r2,
        length=int(rule2_conw)
    )

    # ========================================================
    # Rule 3: single-window overload
    # If in any window:
    # L_k >= tau_D
    # ========================================================
    rule3_flags = (L >= tau_D).astype(np.int8)
    rule3_alerts = int(np.sum(rule3_flags))

    # ========================================================
    # Rule 4: high alert density
    # Generate one Rule-4 alert every time more than eta
    # Rule-3 alerts occur within 60 seconds.
    # Then reset so the same Rule-3 alerts are not reused.
    # ========================================================
    rule4_alerts = _count_rule4_alerts_reset_after_trigger(
        rule3_flags=rule3_flags,
        window_step_s=int(window_step_s),
        eta=int(rule4_eta)
    )

    # ========================================================
    # Rule 5: near-complete blindness
    # If in any window:
    # L_k >= FPS - tau_D
    # ========================================================
    cond_r5 = (L >= rule5_thr)
    rule5_alerts = int(np.sum(cond_r5))

    return {
        "tau_D": int(tau_D),
        "rule5_thr": int(rule5_thr),

        "rule1_alerts": int(rule1_alerts),
        "rule2_alerts": int(rule2_alerts),
        "rule3_alerts": int(rule3_alerts),
        "rule4_alerts": int(rule4_alerts),
        "rule5_alerts": int(rule5_alerts),
    }


# -----------------------------
# Add per-window rule flags
# -----------------------------
def add_rule_flags_to_windows(
    dfw_run: pd.DataFrame,
    fps: float,
    window_s: int,
    num_cameras: int,
    danger_extra: int,
    rule5_late_thr: Optional[int]
) -> pd.DataFrame:

    dfw_run = dfw_run.sort_values("window_idx").reset_index(drop=True).copy()

    tau_D = max(1, (int(num_cameras) + int(danger_extra)) * int(window_s))

    frames_per_window = int(round(float(fps) * float(window_s)))

    if rule5_late_thr is None:
        rule5_thr = frames_per_window - int(tau_D)
    else:
        rule5_thr = int(rule5_late_thr)

    rule5_thr = max(0, rule5_thr)

    L = dfw_run["late_frames"]

    dfw_run["rule1_window_condition"] = ((L >= 2) & (L < tau_D)).astype(int)
    dfw_run["rule2_window_condition"] = (L >= tau_D).astype(int)
    dfw_run["rule3_alert"] = (L >= tau_D).astype(int)
    dfw_run["rule5_alert"] = (L >= rule5_thr).astype(int)

    dfw_run["tau_D"] = int(tau_D)
    dfw_run["rule5_thr"] = int(rule5_thr)

    return dfw_run


# -----------------------------
# Plotting
# -----------------------------
def _plot_timeseries(
    dfw: pd.DataFrame,
    ycol: str,
    out_png: str,
    title: str,
    window_step_s: float
):
    if dfw.empty:
        return

    plt.figure()

    for run_name, g in dfw.groupby("run"):
        g = g.sort_values("window_idx")
        x = g["window_idx"].to_numpy(dtype=float) * float(window_step_s)
        plt.plot(x, g[ycol], label=run_name)

    plt.xlabel("Window start time (s)")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if int(WINDOW_STEP_S) != int(WINDOW_S):
        print("[WARN] WINDOW_STEP_S != WINDOW_S.")
        print("[WARN] Windows overlap, so late frames may be counted multiple times.")
        print("[WARN] For non-overlapping 1-second windows, use WINDOW_STEP_S = WINDOW_S.")

    txts = sorted(
        glob.glob(
            os.path.join(LOG_DIR, "**", TXT_PATTERN),
            recursive=True
        )
    )

    print("LOG_DIR =", LOG_DIR)
    print("OUT_DIR =", OUT_DIR)
    print("Found txt files:", len(txts))

    if not txts:
        raise RuntimeError(f"No txt files found in: {LOG_DIR}")

    runs = [parse_txt_file(p) for p in txts]

    for r in runs:
        print(f"{r.run_name}: parsed rows = {len(r.df)}")

    if all(r.df.empty for r in runs):
        raise RuntimeError("All parsed rows are 0. Regex likely does not match the log lines.")

    # -----------------------------
    # Evaluate windows
    # -----------------------------
    all_rows = []

    for r in runs:
        dfw = eval_windows_detector_fps_budget(
            run=r,
            fps=float(FPS),
            window_s=int(WINDOW_S),
            step_s=int(WINDOW_STEP_S),
            tpost_thr=float(TPOST_THR_MS),
            ttrack_thr=float(TTRACK_THR_MS),
            ttotal_thr=float(TTOTAL_THR_MS)
        )

        if not dfw.empty:
            all_rows.append(dfw)

    if not all_rows:
        raise RuntimeError("No window results produced.")

    all_w = pd.concat(all_rows, ignore_index=True)

    # Add per-window rule flags
    all_w_flagged = []

    for run_name, g in all_w.groupby("run"):
        g2 = add_rule_flags_to_windows(
            dfw_run=g,
            fps=float(FPS),
            window_s=int(WINDOW_S),
            num_cameras=int(NUM_CAMERAS),
            danger_extra=int(DANGER_EXTRA),
            rule5_late_thr=RULE5_LATE_THR
        )
        all_w_flagged.append(g2)

    all_w = pd.concat(all_w_flagged, ignore_index=True)

    out_metrics = os.path.join(
        OUT_DIR,
        f"window_metrics_w{WINDOW_S}.csv"
    )

    all_w.to_csv(out_metrics, index=False)
    print("Saved:", out_metrics)

    # -----------------------------
    # Apply rules per run
    # -----------------------------
    tau_D_global = max(
        1,
        (int(NUM_CAMERAS) + int(DANGER_EXTRA)) * int(WINDOW_S)
    )

    per_run_rows = []

    for run_name, g in all_w.groupby("run"):
        counts = apply_rules_per_run(
            dfw_run=g,
            fps=float(FPS),
            window_s=int(WINDOW_S),
            window_step_s=int(WINDOW_STEP_S),
            rule1_conw=int(RULE1_CONW),
            rule2_conw=int(RULE2_CONW),
            rule4_eta=int(RULE4_ETA),
            num_cameras=int(NUM_CAMERAS),
            danger_extra=int(DANGER_EXTRA),
            rule5_late_thr=RULE5_LATE_THR
        )

        per_run_rows.append({
            "run": run_name,

            "FPS": float(FPS),
            "WINDOW_S": int(WINDOW_S),
            "WINDOW_STEP_S": int(WINDOW_STEP_S),

            "NUM_CAMERAS": int(NUM_CAMERAS),
            "DANGER_EXTRA": int(DANGER_EXTRA),

            "RULE1_CONW": int(RULE1_CONW),
            "RULE2_CONW": int(RULE2_CONW),
            "RULE4_ETA": int(RULE4_ETA),

            "RULE5_LATE_THR": "FPS-tau_D" if RULE5_LATE_THR is None else int(RULE5_LATE_THR),

            **counts
        })

    per_run = pd.DataFrame(per_run_rows).sort_values("run").reset_index(drop=True)

    out_per_run = os.path.join(
        OUT_DIR,
        f"alerts_by_rule_per_run_w{WINDOW_S}.csv"
    )

    per_run.to_csv(out_per_run, index=False)
    print("Saved:", out_per_run)

    # -----------------------------
    # Summary
    # -----------------------------
    total = pd.DataFrame([{
        "LOG_DIR": LOG_DIR,

        "FPS": float(FPS),
        "WINDOW_S": int(WINDOW_S),
        "WINDOW_STEP_S": int(WINDOW_STEP_S),

        "NUM_CAMERAS": int(NUM_CAMERAS),
        "DANGER_EXTRA": int(DANGER_EXTRA),

        "tau_D": int(per_run["tau_D"].iloc[0]) if len(per_run) else int(tau_D_global),

        "RULE1_CONW": int(RULE1_CONW),
        "RULE2_CONW": int(RULE2_CONW),
        "RULE4_ETA": int(RULE4_ETA),

        "RULE5_LATE_THR": "FPS-tau_D" if RULE5_LATE_THR is None else int(RULE5_LATE_THR),
        "rule5_thr": int(per_run["rule5_thr"].iloc[0]) if len(per_run) else -1,

        "n_runs": int(all_w["run"].nunique()),
        "n_windows_total": int(len(all_w)),

        "rule1_alerts_total": int(per_run["rule1_alerts"].sum()),
        "rule2_alerts_total": int(per_run["rule2_alerts"].sum()),
        "rule3_alerts_total": int(per_run["rule3_alerts"].sum()),
        "rule4_alerts_total": int(per_run["rule4_alerts"].sum()),
        "rule5_alerts_total": int(per_run["rule5_alerts"].sum()),
    }])

    out_summary = os.path.join(
        OUT_DIR,
        f"alerts_summary_w{WINDOW_S}.csv"
    )

    total.to_csv(out_summary, index=False)
    print("Saved:", out_summary)

    # -----------------------------
    # Optional plot
    # -----------------------------
    _plot_timeseries(
        dfw=all_w,
        ycol="late_frames",
        out_png=os.path.join(
            OUT_DIR,
            f"w{WINDOW_S}_timeseries_late_frames.png"
        ),
        title=f"w={WINDOW_S}s: late frames per window",
        window_step_s=float(WINDOW_STEP_S)
    )

    print("\nDONE.")
    print("Outputs in:", OUT_DIR)

    print("\nRule settings:")
    print(f"tau_D = {int(tau_D_global)}")
    print(f"Rule 1: 2 <= L_k < tau_D for {RULE1_CONW} consecutive windows")
    print(f"Rule 2: L_k >= tau_D for {RULE2_CONW} consecutive windows")
    print("Rule 3: L_k >= tau_D in any single window")
    print(f"Rule 4: every {RULE4_ETA + 1} Rule-3 alerts within 60 seconds creates one Rule-4 alert, then resets")
    if RULE5_LATE_THR is None:
        print(f"Rule 5: L_k >= FPS - tau_D = {FPS - tau_D_global}")
    else:
        print(f"Rule 5: L_k >= {RULE5_LATE_THR}")


if __name__ == "__main__":
    main()