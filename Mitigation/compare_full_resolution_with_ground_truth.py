import json
import csv
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion



# Folder that contains nuScenes json files:
# attribute.json, calibrated_sensor.json, sample.json, sample_data.json,
# sample_annotation.json, ego_pose.json, category.json, etc.
NUSCENES_META_DIR = Path(
    "/nuscenes/v1.0-trainval_meta/v1.0-trainval"
)

# Folder containing the actual nighttime images fed to YOLO
IMAGE_DIR = Path(
    "images path"
)

# YOLO labels generated at original resolution
YOLO_LABEL_DIR = Path(
    "1920x1080/labels"
)

# Output folder: matched GT objects detected by YOLO at original resolution
OUTPUT_MATCHED_GT_DIR = Path(
    "matched_GT_1920x1080"
)

# Summary CSV
OUTPUT_SUMMARY_CSV = OUTPUT_MATCHED_GT_DIR / "matched_gt_summary.csv"
# ============================================================
# Settings
# ============================================================

IOU_THRESHOLD = 0.5

VALID_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
}

NUSCENES_TO_YOLO = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",

    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
}


# ============================================================
# Loading utilities
# ============================================================

def load_json(name):
    path = NUSCENES_META_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    with open(path, "r") as f:
        return json.load(f)


def make_token_dict(records):
    return {record["token"]: record for record in records}


def group_annotations_by_sample_token(annotations):
    
    ann_tokens_by_sample_token = {}

    for ann in annotations:
        sample_token = ann["sample_token"]
        ann_tokens_by_sample_token.setdefault(sample_token, []).append(ann["token"])

    return ann_tokens_by_sample_token


# ============================================================
# Box and IoU utilities
# ============================================================

def compute_iou(box1, box2):
    """
    box format: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def normalize_yolo_class(name):
    return name.strip().lower()


def load_yolo_labels(label_path):
    """
    Expected YOLO label format:
    class_id class_name xmin ymin xmax ymax confidence

    Example:
    2 car 513 497 563 534 0.3856
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            try:
                class_id = int(parts[0])
                class_name = normalize_yolo_class(parts[1])

                xmin = float(parts[2])
                ymin = float(parts[3])
                xmax = float(parts[4])
                ymax = float(parts[5])
                conf = float(parts[6])
            except ValueError:
                continue

            if class_name not in VALID_CLASSES:
                continue

            boxes.append({
                "class_id": class_id,
                "class_name": class_name,
                "box": [xmin, ymin, xmax, ymax],
                "conf": conf,
            })

    return boxes


def get_image_size(image_stem):
    """
    Finds the image by stem and returns width, height.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = IMAGE_DIR / f"{image_stem}{ext}"

        if image_path.exists():
            img = cv2.imread(str(image_path))

            if img is not None:
                h, w = img.shape[:2]
                return w, h

    # nuScenes camera images are usually 1600x900.
    # This fallback is used only if the image file is not found.
    print(f"Warning: image not found for {image_stem}. Using fallback 1600x900.")
    return 1600, 900


# ============================================================
# Category utility
# ============================================================

def get_annotation_category_name(ann, instance_by_token, category_by_token):
    """
    Supports three possible nuScenes annotation styles:

    1) ann['category_name']
    2) ann['category_token'] -> category.json
    3) ann['instance_token'] -> instance.json -> category.json
    """

    # Case 1: category name is directly in sample_annotation.json
    if "category_name" in ann:
        return ann["category_name"]

    # Case 2: category token is directly in sample_annotation.json
    if "category_token" in ann:
        category_token = ann["category_token"]

        if category_token in category_by_token:
            return category_by_token[category_token].get("name", None)

    # Case 3: category is obtained through instance.json
    if "instance_token" in ann:
        instance_token = ann["instance_token"]

        if instance_token not in instance_by_token:
            return None

        instance = instance_by_token[instance_token]
        category_token = instance.get("category_token", None)

        if category_token is None:
            return None

        if category_token not in category_by_token:
            return None

        return category_by_token[category_token].get("name", None)

    return None


# ============================================================
# nuScenes 3D box projection to 2D
# ============================================================

def box_3d_corners(center, size, rotation):
    """
    Create 3D bounding-box corners in global coordinates.

    nuScenes size format:
    size = [width, length, height]
    """
    width, length, height = size

    x_corners = np.array([
        length / 2, length / 2, -length / 2, -length / 2,
        length / 2, length / 2, -length / 2, -length / 2
    ])

    y_corners = np.array([
        width / 2, -width / 2, -width / 2, width / 2,
        width / 2, -width / 2, -width / 2, width / 2
    ])

    z_corners = np.array([
        height / 2, height / 2, height / 2, height / 2,
        -height / 2, -height / 2, -height / 2, -height / 2
    ])

    corners = np.vstack((x_corners, y_corners, z_corners))

    rotation_matrix = Quaternion(rotation).rotation_matrix
    corners = rotation_matrix @ corners

    corners = corners + np.array(center).reshape(3, 1)

    return corners


def transform_global_to_camera(corners_global, ego_pose, calibrated_sensor):
    """
    Transform 3D corners from global coordinates to camera coordinates.
    """
    # Global -> Ego
    ego_translation = np.array(ego_pose["translation"]).reshape(3, 1)
    ego_rotation = Quaternion(ego_pose["rotation"]).rotation_matrix

    corners_ego = ego_rotation.T @ (corners_global - ego_translation)

    # Ego -> Camera
    sensor_translation = np.array(calibrated_sensor["translation"]).reshape(3, 1)
    sensor_rotation = Quaternion(calibrated_sensor["rotation"]).rotation_matrix

    corners_camera = sensor_rotation.T @ (corners_ego - sensor_translation)

    return corners_camera


def project_camera_box_to_2d(corners_camera, camera_intrinsic, image_w, image_h):
    """
    Project 3D camera-coordinate box corners to a 2D image box.
    """
    camera_intrinsic = np.array(camera_intrinsic)

    depths = corners_camera[2, :]

    # Keep only points in front of the camera.
    valid = depths > 0.1

    if valid.sum() == 0:
        return None

    points = corners_camera[:, valid]

    projected = camera_intrinsic @ points
    projected[:2, :] /= projected[2:3, :]

    xs = projected[0, :]
    ys = projected[1, :]

    xmin = float(np.min(xs))
    ymin = float(np.min(ys))
    xmax = float(np.max(xs))
    ymax = float(np.max(ys))

    # Clip to image boundaries.
    xmin = max(0.0, min(image_w - 1.0, xmin))
    ymin = max(0.0, min(image_h - 1.0, ymin))
    xmax = max(0.0, min(image_w - 1.0, xmax))
    ymax = max(0.0, min(image_h - 1.0, ymax))

    if xmax <= xmin or ymax <= ymin:
        return None

    # Remove very tiny boxes.
    if (xmax - xmin) < 2 or (ymax - ymin) < 2:
        return None

    return [xmin, ymin, xmax, ymax]


def get_gt_2d_boxes_for_image(
    image_stem,
    sample_data_by_filename,
    ann_by_token,
    ann_tokens_by_sample_token,
    ego_pose_by_token,
    calibrated_sensor_by_token,
    instance_by_token,
    category_by_token,
):
    """
    Generate projected 2D GT boxes for one camera image.
    """
    if image_stem not in sample_data_by_filename:
        print(f"Warning: image not found in sample_data.json: {image_stem}")
        return []

    sample_data_record = sample_data_by_filename[image_stem]

    sample_token = sample_data_record["sample_token"]

    ego_pose = ego_pose_by_token[sample_data_record["ego_pose_token"]]
    calibrated_sensor = calibrated_sensor_by_token[
        sample_data_record["calibrated_sensor_token"]
    ]

    camera_intrinsic = calibrated_sensor.get("camera_intrinsic", None)

    if camera_intrinsic is None:
        return []

    image_w, image_h = get_image_size(image_stem)

    gt_boxes = []

    ann_tokens = ann_tokens_by_sample_token.get(sample_token, [])

    for ann_token in ann_tokens:
        ann = ann_by_token[ann_token]

        nusc_category = get_annotation_category_name(
            ann=ann,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
        )

        if nusc_category is None:
            continue

        if nusc_category not in NUSCENES_TO_YOLO:
            continue

        yolo_class = NUSCENES_TO_YOLO[nusc_category]

        if yolo_class not in VALID_CLASSES:
            continue

        corners_global = box_3d_corners(
            center=ann["translation"],
            size=ann["size"],
            rotation=ann["rotation"],
        )

        corners_camera = transform_global_to_camera(
            corners_global=corners_global,
            ego_pose=ego_pose,
            calibrated_sensor=calibrated_sensor,
        )

        box_2d = project_camera_box_to_2d(
            corners_camera=corners_camera,
            camera_intrinsic=camera_intrinsic,
            image_w=image_w,
            image_h=image_h,
        )

        if box_2d is None:
            continue

        gt_boxes.append({
            "class_name": yolo_class,
            "box": box_2d,
            "ann_token": ann_token,
            "nusc_category": nusc_category,
        })

    return gt_boxes


# ============================================================
# Matching YOLO detections to GT
# ============================================================

def match_yolo_to_gt(gt_boxes, yolo_boxes, iou_threshold=0.5):
    """
    One-to-one greedy matching by highest IoU.
    Returns matched GT records.
    """
    candidate_pairs = []

    for gt_idx, gt in enumerate(gt_boxes):
        for det_idx, det in enumerate(yolo_boxes):
            if gt["class_name"] != det["class_name"]:
                continue

            iou = compute_iou(gt["box"], det["box"])

            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_idx, det_idx))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_det = set()
    matches = []

    for iou, gt_idx, det_idx in candidate_pairs:
        if gt_idx in used_gt or det_idx in used_det:
            continue

        used_gt.add(gt_idx)
        used_det.add(det_idx)

        gt = gt_boxes[gt_idx]
        det = yolo_boxes[det_idx]

        matches.append({
            "class_name": gt["class_name"],
            "gt_box": gt["box"],
            "ann_token": gt["ann_token"],
            "nusc_category": gt["nusc_category"],
            "matched_yolo_box": det["box"],
            "matched_yolo_conf": det["conf"],
            "iou": iou,
        })

    return matches


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_MATCHED_GT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes metadata...")

    sample_data = load_json("sample_data.json")
    annotations = load_json("sample_annotation.json")
    ego_poses = load_json("ego_pose.json")
    calibrated_sensors = load_json("calibrated_sensor.json")
    instances = load_json("instance.json")
    categories = load_json("category.json")

    ann_by_token = make_token_dict(annotations)
    ego_pose_by_token = make_token_dict(ego_poses)
    calibrated_sensor_by_token = make_token_dict(calibrated_sensors)
    instance_by_token = make_token_dict(instances)
    category_by_token = make_token_dict(categories)

    ann_tokens_by_sample_token = group_annotations_by_sample_token(annotations)

    # Debug: useful for checking metadata format.
    if len(annotations) > 0:
        print("Example annotation keys:", list(annotations[0].keys()))
    if len(instances) > 0:
        print("Example instance keys:", list(instances[0].keys()))
    if len(categories) > 0:
        print("Example category keys:", list(categories[0].keys()))

    # Map image filename stem to sample_data record.
    # sample_data filename looks like:
    # samples/CAM_BACK/n015-...__CAM_BACK__1542193081537525.jpg
    sample_data_by_filename = {}

    for sd in sample_data:
        filename = Path(sd["filename"]).name
        stem = Path(filename).stem

        # Keep only camera images.
        if "CAM_" in filename:
            sample_data_by_filename[stem] = sd

    yolo_label_files = sorted(YOLO_LABEL_DIR.glob("*.txt"))

    print(f"Found YOLO label files: {len(yolo_label_files)}")

    summary_rows = []

    total_gt = 0
    total_yolo = 0
    total_matched = 0
    total_no_gt_image = 0
    total_not_in_sample_data = 0

    for label_path in yolo_label_files:
        image_stem = label_path.stem

        if image_stem not in sample_data_by_filename:
            total_not_in_sample_data += 1

        yolo_boxes = load_yolo_labels(label_path)

        gt_boxes = get_gt_2d_boxes_for_image(
            image_stem=image_stem,
            sample_data_by_filename=sample_data_by_filename,
            ann_by_token=ann_by_token,
            ann_tokens_by_sample_token=ann_tokens_by_sample_token,
            ego_pose_by_token=ego_pose_by_token,
            calibrated_sensor_by_token=calibrated_sensor_by_token,
            instance_by_token=instance_by_token,
            category_by_token=category_by_token,
        )

        if len(gt_boxes) == 0:
            total_no_gt_image += 1

        matches = match_yolo_to_gt(
            gt_boxes=gt_boxes,
            yolo_boxes=yolo_boxes,
            iou_threshold=IOU_THRESHOLD,
        )

        # Save matched GT boxes.
        # This is the reference set:
        # GT objects that YOLO detected at original resolution.
        out_txt = OUTPUT_MATCHED_GT_DIR / f"{image_stem}.txt"

        with open(out_txt, "w") as f:
            for m in matches:
                xmin, ymin, xmax, ymax = m["gt_box"]

                f.write(
                    f"{m['class_name']} "
                    f"{int(round(xmin))} {int(round(ymin))} "
                    f"{int(round(xmax))} {int(round(ymax))} "
                    f"{m['matched_yolo_conf']:.4f} "
                    f"{m['iou']:.4f} "
                    f"{m['ann_token']}\n"
                )

        total_gt += len(gt_boxes)
        total_yolo += len(yolo_boxes)
        total_matched += len(matches)

        summary_rows.append({
            "image": image_stem,
            "gt_boxes": len(gt_boxes),
            "yolo_boxes": len(yolo_boxes),
            "matched_gt": len(matches),
            "output_file": str(out_txt),
        })

        print(
            f"{image_stem}: "
            f"GT={len(gt_boxes)}, "
            f"YOLO={len(yolo_boxes)}, "
            f"matched={len(matches)}"
        )

    # Save summary CSV.
    with open(OUTPUT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "gt_boxes",
                "yolo_boxes",
                "matched_gt",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDone.")
    print("=" * 60)
    print(f"Total YOLO label files: {len(yolo_label_files)}")
    print(f"Total projected GT boxes: {total_gt}")
    print(f"Total YOLO boxes: {total_yolo}")
    print(f"Total matched GT boxes: {total_matched}")
    print(f"Images with zero projected GT boxes: {total_no_gt_image}")
    print(f"YOLO images not found in sample_data.json: {total_not_in_sample_data}")
    print(f"Matched GT folder: {OUTPUT_MATCHED_GT_DIR}")
    print(f"Summary CSV: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()