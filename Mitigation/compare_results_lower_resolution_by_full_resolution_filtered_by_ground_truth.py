import csv
from pathlib import Path


YOLO_ROOT = Path(
    "Root path that includes lower resolution folders, for exmaple, res_1280*704 or 960*512 "
)

# The folder generated from comparing 1920x1080 YOLO with GT
REFERENCE_GT_DIR = Path(
    "original_1920x1080/matched_gt_yolo_1920x1080"
)

OUTPUT_CSV = YOLO_ROOT / "lower_resolution_vs_kept_gt_comparison.csv"

#Save per-frame retained/missed object lists
OUTPUT_DETAIL_DIR = YOLO_ROOT / "lower_resolution_vs_kept_gt_details"


# ============================================================
# Settings
# ============================================================

IOU_THRESHOLD = 0.5

LOWER_RESOLUTIONS = [
    "1280x704",
    "960x512",
]

VALID_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
}


# ============================================================
# Utilities
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


def normalize_class_name(class_name):
    return class_name.strip().lower()


def load_reference_gt(reference_path):
    """
    Expected format:
    class_name xmin ymin xmax ymax matched_yolo_conf iou ann_token

    Returns list:
    {
        "class_name": str,
        "box": [xmin, ymin, xmax, ymax],
        "orig_conf": float,
        "orig_iou": float,
        "ann_token": str
    }
    """
    objects = []

    if not reference_path.exists():
        return objects

    with open(reference_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 8:
                continue

            try:
                class_name = normalize_class_name(parts[0])

                xmin = float(parts[1])
                ymin = float(parts[2])
                xmax = float(parts[3])
                ymax = float(parts[4])

                orig_conf = float(parts[5])
                orig_iou = float(parts[6])
                ann_token = parts[7]

            except ValueError:
                continue

            if class_name not in VALID_CLASSES:
                continue

            objects.append({
                "class_name": class_name,
                "box": [xmin, ymin, xmax, ymax],
                "orig_conf": orig_conf,
                "orig_iou": orig_iou,
                "ann_token": ann_token,
            })

    return objects


def load_yolo_labels(label_path):
    """
    Expected format:
    class_id class_name xmin ymin xmax ymax confidence

    Returns list:
    {
        "class_id": int,
        "class_name": str,
        "box": [xmin, ymin, xmax, ymax],
        "conf": float
    }
    """
    detections = []

    if not label_path.exists():
        return detections

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            try:
                class_id = int(parts[0])
                class_name = normalize_class_name(parts[1])

                xmin = float(parts[2])
                ymin = float(parts[3])
                xmax = float(parts[4])
                ymax = float(parts[5])

                conf = float(parts[6])

            except ValueError:
                continue

            if class_name not in VALID_CLASSES:
                continue

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "box": [xmin, ymin, xmax, ymax],
                "conf": conf,
            })

    return detections


def match_reference_to_detections(reference_objects, detections, iou_threshold=0.5):
    """
    One-to-one greedy matching.
    A kept/reference object is retained if a lower-resolution detection
    with the same class has IoU >= threshold.

    Returns:
    matched_reference_indices, matches
    """
    candidate_pairs = []

    for ref_idx, ref in enumerate(reference_objects):
        for det_idx, det in enumerate(detections):
            if ref["class_name"] != det["class_name"]:
                continue

            iou = compute_iou(ref["box"], det["box"])

            if iou >= iou_threshold:
                candidate_pairs.append((iou, ref_idx, det_idx))

    candidate_pairs.sort(reverse=True, key=lambda x: x[0])

    used_refs = set()
    used_dets = set()
    matches = []

    for iou, ref_idx, det_idx in candidate_pairs:
        if ref_idx in used_refs or det_idx in used_dets:
            continue

        used_refs.add(ref_idx)
        used_dets.add(det_idx)

        ref = reference_objects[ref_idx]
        det = detections[det_idx]

        matches.append({
            "ref_idx": ref_idx,
            "det_idx": det_idx,
            "class_name": ref["class_name"],
            "ann_token": ref["ann_token"],
            "iou": iou,
            "lower_conf": det["conf"],
            "orig_conf": ref["orig_conf"],
            "orig_iou": ref["orig_iou"],
        })

    return used_refs, matches


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    reference_files = sorted(REFERENCE_GT_DIR.glob("*.txt"))

    print(f"Reference files found: {len(reference_files)}")

    rows = []

    # For global summary
    summary = {
        res: {
            "frames": 0,
            "reference_objects": 0,
            "retained_objects": 0,
            "missed_objects": 0,
            "lower_yolo_detections": 0,
        }
        for res in LOWER_RESOLUTIONS
    }

    for ref_path in reference_files:
        image_stem = ref_path.stem
        reference_objects = load_reference_gt(ref_path)

        # Skip frames with no kept objects from original resolution.
        if len(reference_objects) == 0:
            continue

        for res in LOWER_RESOLUTIONS:
            lower_label_path = YOLO_ROOT / res / "labels" / f"{image_stem}.txt"
            detections = load_yolo_labels(lower_label_path)

            matched_ref_indices, matches = match_reference_to_detections(
                reference_objects=reference_objects,
                detections=detections,
                iou_threshold=IOU_THRESHOLD,
            )

            retained = len(matched_ref_indices)
            total_ref = len(reference_objects)
            missed = total_ref - retained
            retention_percent = retained / total_ref * 100 if total_ref > 0 else 0.0

            summary[res]["frames"] += 1
            summary[res]["reference_objects"] += total_ref
            summary[res]["retained_objects"] += retained
            summary[res]["missed_objects"] += missed
            summary[res]["lower_yolo_detections"] += len(detections)

            rows.append({
                "image": image_stem,
                "resolution": res,
                "reference_objects": total_ref,
                "lower_yolo_detections": len(detections),
                "retained_objects": retained,
                "missed_objects": missed,
                "retention_percent": retention_percent,
            })

            # Save per-frame retained and missed details
            detail_res_dir = OUTPUT_DETAIL_DIR / res
            detail_res_dir.mkdir(parents=True, exist_ok=True)

            retained_path = detail_res_dir / f"{image_stem}_retained.txt"
            missed_path = detail_res_dir / f"{image_stem}_missed.txt"

            with open(retained_path, "w") as f_retained:
                for m in matches:
                    f_retained.write(
                        f"{m['class_name']} "
                        f"{m['ann_token']} "
                        f"lower_conf={m['lower_conf']:.4f} "
                        f"orig_conf={m['orig_conf']:.4f} "
                        f"iou_lower_ref={m['iou']:.4f} "
                        f"orig_iou={m['orig_iou']:.4f}\n"
                    )

            with open(missed_path, "w") as f_missed:
                for idx, ref in enumerate(reference_objects):
                    if idx in matched_ref_indices:
                        continue

                    xmin, ymin, xmax, ymax = ref["box"]

                    f_missed.write(
                        f"{ref['class_name']} "
                        f"{int(round(xmin))} {int(round(ymin))} "
                        f"{int(round(xmax))} {int(round(ymax))} "
                        f"orig_conf={ref['orig_conf']:.4f} "
                        f"orig_iou={ref['orig_iou']:.4f} "
                        f"{ref['ann_token']}\n"
                    )

    # Save per-image CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "resolution",
                "reference_objects",
                "lower_yolo_detections",
                "retained_objects",
                "missed_objects",
                "retention_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved per-image comparison CSV:")
    print(OUTPUT_CSV)

    print("\nSummary by resolution")
    print("=" * 70)

    for res in LOWER_RESOLUTIONS:
        s = summary[res]

        total_ref = s["reference_objects"]
        retained = s["retained_objects"]
        missed = s["missed_objects"]

        retention_percent = retained / total_ref * 100 if total_ref > 0 else 0.0
        missed_percent = missed / total_ref * 100 if total_ref > 0 else 0.0

        print(f"\nResolution: {res}")
        print(f"Frames evaluated: {s['frames']}")
        print(f"Reference objects from 1920x1080: {total_ref}")
        print(f"Lower-resolution YOLO detections: {s['lower_yolo_detections']}")
        print(f"Retained objects: {retained}")
        print(f"Missed objects: {missed}")
        print(f"Retention: {retention_percent:.2f}%")
        print(f"Missed: {missed_percent:.2f}%")

    print("\nDetail files saved in:")
    print(OUTPUT_DETAIL_DIR)


if __name__ == "__main__":
    main()