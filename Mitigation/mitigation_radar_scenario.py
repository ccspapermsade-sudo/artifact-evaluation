from pathlib import Path
import re
import numpy as np
import pandas as pd
from nuscenes.utils.data_classes import RadarPointCloud

# =========================
# Change these paths
# =========================
IMAGE_DIR = Path("image path")
NUSCENES_ROOT = Path(".../nuscenes")

# =========================
# Safety gate settings
# =========================
DIST_THRESHOLD_M = 10.0

# =========================
# Scenario setting
# =========================
# SCENARIO = 1: use CAM_FRONT and CAM_FRONT_RIGHT
# SCENARIO = 2: use CAM_FRONT, CAM_FRONT_RIGHT, and CAM_BACK_RIGHT
SCENARIO = 1

SCENARIO_CAMERAS = {
    1: {"CAM_FRONT", "CAM_FRONT_RIGHT"},
    2: {"CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"},
}

# Front region around ego vehicle
FRONT_Y_MIN = -3.0
FRONT_Y_MAX = 3.0
FRONT_X_MIN = 0.0
FRONT_X_MAX = 30.0

# Right-side region around ego vehicle
RIGHT_Y_MIN = -6.0
RIGHT_Y_MAX = 0.0
X_MIN = -10.0
X_MAX = 30.0

CAM_TO_RADAR = {
    "CAM_FRONT": "RADAR_FRONT",
    "CAM_FRONT_LEFT": "RADAR_FRONT_LEFT",
    "CAM_FRONT_RIGHT": "RADAR_FRONT_RIGHT",
    "CAM_BACK_LEFT": "RADAR_BACK_LEFT",
    "CAM_BACK_RIGHT": "RADAR_BACK_RIGHT",
    "CAM_BACK": "RADAR_BACK_LEFT",
}

timestamp_pattern = re.compile(r"__(CAM_[A-Z_]+)__(\d+)")


def parse_image_name(image_path):
    match = timestamp_pattern.search(image_path.stem)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def get_radar_files_all_blobs(radar_name):
    files = []

    search_pattern = f"v1.0-trainval*_blobs/samples/{radar_name}"

    for radar_dir in NUSCENES_ROOT.glob(search_pattern):
        if not radar_dir.exists():
            continue

        for pcd_file in radar_dir.glob("*.pcd"):
            nums = re.findall(r"\d+", pcd_file.stem)
            if nums:
                timestamp = int(nums[-1])
                files.append((timestamp, pcd_file))

    return sorted(files)


def find_closest_radar_file(radar_files, image_timestamp):
    if not radar_files:
        return None, None

    timestamps = np.array([t for t, _ in radar_files])
    idx = np.argmin(np.abs(timestamps - image_timestamp))

    radar_timestamp, radar_file = radar_files[idx]
    time_diff_us = abs(radar_timestamp - image_timestamp)

    return radar_file, time_diff_us


def nearest_region_distance(pcd_path, region_name):
    radar_pc = RadarPointCloud.from_file(str(pcd_path))
    points = radar_pc.points

    if points.shape[1] == 0:
        return None, 0, 0

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    if region_name == "front":
        region_mask = (
            (y >= FRONT_Y_MIN) &
            (y <= FRONT_Y_MAX) &
            (x >= FRONT_X_MIN) &
            (x <= FRONT_X_MAX)
        )
    elif region_name == "right":
        region_mask = (
            (y >= RIGHT_Y_MIN) &
            (y <= RIGHT_Y_MAX) &
            (x >= X_MIN) &
            (x <= X_MAX)
        )
    else:
        raise ValueError(f"Unknown region_name: {region_name}")

    total_points = points.shape[1]
    region_points = int(np.sum(region_mask))

    if region_points == 0:
        return None, int(total_points), 0

    distances = np.sqrt(
        x[region_mask] ** 2 +
        y[region_mask] ** 2 +
        z[region_mask] ** 2
    )

    min_distance = float(np.min(distances))

    return min_distance, int(total_points), region_points


def region_for_camera(cam_name):
    if cam_name == "CAM_FRONT":
        return "front"
    if cam_name in {"CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"}:
        return "right"
    return None


# =========================
# Preload radar files
# =========================
radar_cache = {}

for radar_name in set(CAM_TO_RADAR.values()):
    radar_cache[radar_name] = get_radar_files_all_blobs(radar_name)
    print(f"{radar_name}: found {len(radar_cache[radar_name])} radar files")


# =========================
# Process masked images
# =========================
results = []

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print(f"\nFound {len(image_files)} masked images.\n")

if SCENARIO not in SCENARIO_CAMERAS:
    raise ValueError(f"SCENARIO must be 1 or 2, got: {SCENARIO}")

allowed_cameras = SCENARIO_CAMERAS[SCENARIO]
print(f"Using SCENARIO={SCENARIO}, allowed cameras={sorted(allowed_cameras)}\n")

for image_path in image_files:
    cam_name, image_timestamp = parse_image_name(image_path)

    if cam_name is None:
        print(f"Skipping: cannot parse filename: {image_path.name}")
        continue

    if cam_name not in allowed_cameras:
        continue

    region_name = region_for_camera(cam_name)
    if region_name is None:
        print(f"Skipping: no region rule for {cam_name}")
        continue

    radar_name = CAM_TO_RADAR.get(cam_name)

    if radar_name is None:
        print(f"Skipping: no radar mapping for {cam_name}")
        continue

    radar_files = radar_cache.get(radar_name, [])

    radar_file, time_diff_us = find_closest_radar_file(
        radar_files,
        image_timestamp
    )

    if radar_file is None:
        print(f"No radar file found for {image_path.name}")
        continue

    min_dist, total_points, region_points = nearest_region_distance(radar_file, region_name)

    if min_dist is None:
        close_object = False
        min_dist_print = "empty"
    else:
        close_object = min_dist <= DIST_THRESHOLD_M
        min_dist_print = f"{min_dist:.2f}"

    decision = "BLOCK" if close_object else "ALLOW"

    results.append({
        "image": image_path.name,
        "camera": cam_name,
        "image_timestamp": image_timestamp,
        "scenario": SCENARIO,
        "region": region_name,
        "radar": radar_name,
        "radar_file": radar_file.name,
        "radar_path": str(radar_file),
        "time_diff_us": time_diff_us,
        "total_radar_points": total_points,
        "region_radar_points": region_points,
        "nearest_region_distance_m": min_dist,
        "threshold_m": DIST_THRESHOLD_M,
        "region_close_object": close_object,
        "lane_change_decision": decision,
    })

    print(
        f"{image_path.name} | {cam_name}->{radar_name} | "
        f"region={region_name} | "
        f"nearest_{region_name}={min_dist_print} m | "
        f"region_points={region_points} | "
        f"time_diff={time_diff_us} us | "
        f"decision={decision}"
    )


# =========================
# Save CSV
# =========================
df = pd.DataFrame(results)

out_csv = IMAGE_DIR / f"scenario_{SCENARIO}_radar_safety_gate_results.csv"
df.to_csv(out_csv, index=False)

print(f"\nSaved results to:\n{out_csv}")


# =========================
# Summary
# =========================
if len(df) > 0:
    blocked = (df["lane_change_decision"] == "BLOCK").sum()
    allowed = (df["lane_change_decision"] == "ALLOW").sum()

    print("\nSummary:")
    print(f"Total matched masked images: {len(df)}")
    print(f"Blocked lane changes: {blocked}")
    print(f"Allowed lane changes: {allowed}")
    print(f"Blocked ratio: {blocked / len(df) * 100:.2f}%")

    valid_dist = df["nearest_region_distance_m"].dropna()

    if len(valid_dist) > 0:
        print(f"Minimum nearest region distance: {valid_dist.min():.2f} m")
        print(f"Mean nearest region distance: {valid_dist.mean():.2f} m")
        print(f"Median nearest region distance: {valid_dist.median():.2f} m")