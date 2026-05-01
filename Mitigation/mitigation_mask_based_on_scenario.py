#!/usr/bin/env python3
import os
import glob
import csv
import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

# =========================================================
# PARAMETERS (EDIT THESE)
# =========================================================
IMG_DIR = "imagepath"
OUT_DIR = "Output path to save masked_images"

# Scenario selection:
# 1 -> only FRONT and FRONT_RIGHT images
# 2 -> FRONT, FRONT_RIGHT, and BACK_RIGHT images
SCENARIO = 1

# nuScenes MINI
NUSC_DATAROOT = "../nuscenes/v1.0-trainval_meta"
NUSC_VERSION = "v1.0-trainval"
# Geometry
PPM = 35.0               # Pixels Per Meter
CAR_LENGTH_M = 4.5        # meters (minimum distance coverage)
VEHICLE_WIDTH_PX = 500    # Type-1 FRONT ROI width (vehicle width in pixels)

# Type-2 (front-right):
# - ROI height FIXED = CAR_WIDTH_PX
# - ROI length (horizontal) relative to speed via h_px
CAR_WIDTH_PX = 400

# Mask color for the covered part (OpenCV uses BGR)
MASK_COLOR = (0,0,0)   # white; black would be (0,0,0)

# Saving settings to avoid extra preprocess time (main knobs)
SAVE_AS_JPEG = True
JPG_QUALITY = 85        # try 80–90 (lower => smaller => often faster to load/decode)
JPG_OPTIMIZE = True     # can reduce size; encoding is slower but one-time offline
JPG_PROGRESSIVE = False # progressive JPEG can decode slower in some pipelines; keep False

# =========================================================


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def camera_type_from_filename(path: str):
    """Return front/front_right/back_right if the camera name is in the filename."""
    base = os.path.basename(path).lower().replace("-", "_")

    # Check the longer names first because "front_right" also contains "front".
    if "front_right" in base or "cam_front_right" in base:
        return "front_right"
    if "back_right" in base or "cam_back_right" in base:
        return "back_right"
    if "front" in base or "cam_front" in base:
        return "front"
    return None


def allowed_camera_types_for_scenario(scenario: int):
    if scenario == 1:
        return {"front", "front_right"}
    if scenario == 2:
        return {"front", "front_right", "back_right"}
    raise ValueError("SCENARIO must be 1 or 2.")


def sorted_image_list(img_dir: str) -> list:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))

    allowed = allowed_camera_types_for_scenario(SCENARIO)
    files = [f for f in files if camera_type_from_filename(f) in allowed]
    files.sort()

    if not files:
        raise FileNotFoundError(
            f"No images found in: {img_dir} for SCENARIO={SCENARIO}, allowed cameras={sorted(allowed)}"
        )

    print(f"[INFO] SCENARIO={SCENARIO}, allowed cameras={sorted(allowed)}")
    print(f"[INFO] Found {len(files)} matching images")
    return files


def clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(int(round(x)), hi)))


def mask_keep_rect_color_else(img_bgr, x0, y0, x1, y1, mask_color=(255, 255, 255)):
    """Keep ROI, paint everything else mask_color."""
    h, w = img_bgr.shape[:2]
    x0 = clamp_int(x0, 0, w)
    x1 = clamp_int(x1, 0, w)
    y0 = clamp_int(y0, 0, h)
    y1 = clamp_int(y1, 0, h)

    out = np.empty_like(img_bgr)
    out[:] = mask_color

    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = img_bgr[y0:y1, x0:x1]
    return out


def compute_h_px_from_speed(v_mps: float) -> int:
    """
    d_m = max(v*1s, car_length_m)
    h_px = d_m * PPM

    h_px is used as:
      - Type-1 FRONT: vertical extent
      - Type-2 FRONT-RIGHT: horizontal extent (length)
    """
    d_m = max(v_mps * 1.0, CAR_LENGTH_M)
    return max(1, int(round(d_m * PPM)))


def build_basename_to_sample_data_token(nusc: NuScenes) -> dict:
    """basename(image.jpg) -> sample_data_token"""
    m = {}
    for sd in nusc.sample_data:
        fn = sd.get("filename", "")
        if fn:
            base = os.path.basename(fn)
            m[base] = sd["token"]
    return m


def get_scene_name_and_timestamp_for_image(nusc: NuScenes, sd_token: str):
    """sample_data_token -> (scene_name, timestamp_us)"""
    sd = nusc.get("sample_data", sd_token)
    ts = int(sd["timestamp"])  # microseconds
    sample = nusc.get("sample", sd["sample_token"])
    scene = nusc.get("scene", sample["scene_token"])
    return scene["name"], ts


def speed_from_can_bus(nusc_can: NuScenesCanBus, scene_name: str, timestamp_us: int) -> float:
    """Closest CAN speed message to timestamp_us."""
    candidates = [
        ("vehicle_monitor", ["vehicle_speed", "speed", "vel", "velocity"]),
        ("zoe_veh_info",    ["vehicle_speed", "speed", "vel", "velocity"]),
        ("pose",            ["speed", "vel", "velocity"]),
    ]

    best = None  # (abs_dt_us, speed)

    for msg_name, keys in candidates:
        try:
            msgs = nusc_can.get_messages(scene_name, msg_name)
        except Exception:
            continue
        if not msgs:
            continue

        for m in msgs:
            utime = m.get("utime", None)
            if utime is None:
                continue
            dt = abs(int(utime) - int(timestamp_us))
            for k in keys:
                if k in m and m[k] is not None:
                    try:
                        sp = float(m[k])
                        if best is None or dt < best[0]:
                            best = (dt, sp)
                    except Exception:
                        pass

    if best is None:
        raise RuntimeError("Could not find CAN bus speed (missing CAN data or key mismatch).")

    return float(best[1])


def save_image_fast(out_path: str, img_bgr: np.ndarray) -> None:
    """
    Save masked image with controlled encoding to reduce I/O + decode cost later.
    We save as JPEG with fixed settings for consistency.
    """
    ensure_dir(os.path.dirname(out_path))

    if SAVE_AS_JPEG:
        # Force .jpg extension
        root, _ = os.path.splitext(out_path)
        out_path = root + ".jpg"

        params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY),
        ]
        # Optional flags (may not be supported in very old OpenCV builds)
        if hasattr(cv2, "IMWRITE_JPEG_OPTIMIZE"):
            params += [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1 if JPG_OPTIMIZE else 0]
        if hasattr(cv2, "IMWRITE_JPEG_PROGRESSIVE"):
            params += [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1 if JPG_PROGRESSIVE else 0]

        ok, buf = cv2.imencode(".jpg", img_bgr, params)
        if not ok:
            raise RuntimeError(f"Failed to encode JPEG for: {out_path}")
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        return

    # SAVE_AS_JPEG=False, fallback to OpenCV default
    cv2.imwrite(out_path, img_bgr)


def main():
    ensure_dir(OUT_DIR)
    files = sorted_image_list(IMG_DIR)

    # Load nuScenes + CAN bus
    nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_DATAROOT, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=NUSC_DATAROOT)

    base2sd = build_basename_to_sample_data_token(nusc)

    manifest_path = os.path.join(OUT_DIR, "masked_manifest.csv")
    rows = []

    for img_i, img_path in enumerate(files, start=1):
        cam_type = camera_type_from_filename(img_path)
        if cam_type is None:
            continue

        img_base = os.path.basename(img_path)
        if img_base not in base2sd:
            raise RuntimeError(f"Could not map image to nuScenes sample_data: {img_base}")

        sd_token = base2sd[img_base]
        scene_name, ts_us = get_scene_name_and_timestamp_for_image(nusc, sd_token)

        v = speed_from_can_bus(nusc_can, scene_name, ts_us)
        h_px = compute_h_px_from_speed(v)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        H, W = img.shape[:2]

        # =========================
        # TYPE 1: FRONT
        # - width fixed = VEHICLE_WIDTH_PX
        # - height dynamic = h_px
        # =========================
        if cam_type == "front":
            roi_h = min(h_px, H)
            cx = W // 2
            half_w = VEHICLE_WIDTH_PX // 2
            x0 = cx - half_w
            x1 = cx + (VEHICLE_WIDTH_PX - half_w)
            y1 = H
            y0 = H - roi_h

        # =========================
        # TYPE 2: FRONT-RIGHT / BACK-RIGHT
        # - height FIXED = CAR_WIDTH_PX
        # - length relative to speed => ROI WIDTH = h_px
        # bottom-left ROI
        # =========================
        elif cam_type in {"front_right", "back_right"}:
            roi_h = min(CAR_WIDTH_PX, H)  # fixed thickness
            roi_w = min(h_px, W)          # speed-based length

            x0, x1 = 0, roi_w
            y1 = H
            y0 = H - roi_h

        else:
            continue

        masked = mask_keep_rect_color_else(img, x0, y0, x1, y1, MASK_COLOR)

        out_name = f"{img_i:06d}_{cam_type}_{os.path.splitext(img_base)[0]}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        save_image_fast(out_path, masked)

        out_size = os.path.getsize(os.path.splitext(out_path)[0] + ".jpg")
        in_size = os.path.getsize(img_path)

        rows.append([
            img_i, scene_name, cam_type,
            img_base, os.path.basename(os.path.splitext(out_path)[0] + ".jpg"),
            v, h_px, ts_us,
            (x1 - x0), (y1 - y0),
            str(MASK_COLOR),
            in_size, out_size, JPG_QUALITY
        ])

        if img_i % 100 == 0:
            print(f"[INFO] Processed {img_i}/{len(files)} images")

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_index", "scene_name", "cam_type",
            "in_file", "out_file",
            "speed_mps", "h_px(speed_extent_px)", "timestamp_us",
            "roi_w_px", "roi_h_px",
            "mask_color_bgr",
            "in_bytes", "out_bytes", "jpg_quality"
        ])
        w.writerows(rows)

    print(f"[DONE] Masked images saved to: {OUT_DIR}")
    print(f"[DONE] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
