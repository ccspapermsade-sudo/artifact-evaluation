import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# -------------------------------
# Load YOLOv5 model using Ultralytics
# -------------------------------
# local yolov5s.pt path
# Example:
# model = YOLO(".../yolov5/yolov5/yolov5s.pt")
model = YOLO("yolov5s.pt")


def process_images(
    input_directory,
    output_root_directory,
    log_file,
    resolutions,
    conf_thres=0.25,
    iou_thres=0.45,
):
    input_directory = Path(input_directory)
    output_root_directory = Path(output_root_directory)
    output_root_directory.mkdir(parents=True, exist_ok=True)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png"}

    total_time = 0.0

    with open(log_file, "w") as f:
        f.write("Ultralytics YOLOv5 Image Processing Times\n")
        f.write("=" * 60 + "\n\n")
        f.write("Saved label format:\n")
        f.write("class_id class_name xmin ymin xmax ymax confidence\n")
        f.write("Bounding boxes are mapped back to the original image coordinates.\n")
        f.write("=" * 60 + "\n\n")

        image_paths = [
            p for p in sorted(input_directory.iterdir())
            if p.suffix.lower() in image_extensions
        ]

        for res_name, (target_w, target_h) in resolutions.items():
            f.write(f"\nResolution: {res_name} ({target_w}x{target_h})\n")
            f.write("-" * 60 + "\n")

            # Each resolution gets its own folder.
            output_image_directory = output_root_directory / res_name / "images"
            output_text_directory = output_root_directory / res_name / "labels"

            output_image_directory.mkdir(parents=True, exist_ok=True)
            output_text_directory.mkdir(parents=True, exist_ok=True)

            resolution_total_time = 0.0
            num_images = 0
            total_detections = 0

            for image_path in image_paths:
                num_images += 1
                print(f"Processing {image_path.name} at {res_name}")

                try:
                    img_bgr = cv2.imread(str(image_path))
                    if img_bgr is None:
                        raise ValueError(f"Could not read image: {image_path}")

                    original_h, original_w = img_bgr.shape[:2]

                    # Resize image to the tested resolution
                    resized_bgr = cv2.resize(img_bgr, (target_w, target_h))

                    start_time = time.time()

                    # Run Ultralytics YOLO
                    results = model.predict(
                        source=resized_bgr,
                        conf=conf_thres,
                        iou=iou_thres,
                        verbose=False
                    )

                    end_time = time.time()
                    processing_time = end_time - start_time

                    total_time += processing_time
                    resolution_total_time += processing_time

                    result = results[0]

                    # Save visualized detection image
                    plotted_bgr = result.plot()  # BGR image
                    output_image_path = output_image_directory / image_path.name
                    cv2.imwrite(str(output_image_path), plotted_bgr)

                    # Save detection labels mapped back to original image coordinates
                    text_file_path = output_text_directory / f"{image_path.stem}.txt"

                    scale_x = original_w / target_w
                    scale_y = original_h / target_h

                    boxes = result.boxes

                    with open(text_file_path, "w") as txt_file:
                        if boxes is not None and len(boxes) > 0:
                            xyxy = boxes.xyxy.cpu().numpy()
                            confs = boxes.conf.cpu().numpy()
                            clss = boxes.cls.cpu().numpy().astype(int)

                            total_detections += len(xyxy)

                            for box, conf, cls_id in zip(xyxy, confs, clss):
                                xmin, ymin, xmax, ymax = box

                                # Map resized-image coordinates back to original image coordinates
                                xmin_orig = xmin * scale_x
                                xmax_orig = xmax * scale_x
                                ymin_orig = ymin * scale_y
                                ymax_orig = ymax * scale_y

                                # Clip to original image bounds
                                xmin_orig = max(0, min(original_w - 1, xmin_orig))
                                xmax_orig = max(0, min(original_w - 1, xmax_orig))
                                ymin_orig = max(0, min(original_h - 1, ymin_orig))
                                ymax_orig = max(0, min(original_h - 1, ymax_orig))

                                class_name = model.names[int(cls_id)]

                                txt_file.write(
                                    f"{int(cls_id)} {class_name} "
                                    f"{int(xmin_orig)} {int(ymin_orig)} "
                                    f"{int(xmax_orig)} {int(ymax_orig)} "
                                    f"{float(conf):.4f}\n"
                                )

                    f.write(
                        f"{image_path.name}: "
                        f"{processing_time:.4f} seconds, "
                        f"detections={len(boxes) if boxes is not None else 0}\n"
                    )

                except Exception as e:
                    error_msg = f"Error processing {image_path.name}: {str(e)}"
                    f.write(error_msg + "\n")
                    print(error_msg)

            avg_time = resolution_total_time / num_images if num_images > 0 else 0.0
            avg_det = total_detections / num_images if num_images > 0 else 0.0

            f.write(f"\nImages processed at {res_name}: {num_images}\n")
            f.write(f"Total detections at {res_name}: {total_detections}\n")
            f.write(f"Average detections/image at {res_name}: {avg_det:.2f}\n")
            f.write(f"Total time for {res_name}: {resolution_total_time:.4f} seconds\n")
            f.write(f"Average time for {res_name}: {avg_time:.4f} seconds/image\n")
            f.write("-" * 60 + "\n")

        f.write(f"\nOverall total processing time: {total_time:.4f} seconds\n")

    print(f"Overall total processing time: {total_time:.4f} seconds")


# -------------------------------
# Directories
# -------------------------------
input_directory = "image path"

output_root_directory = "Output path"

log_file = "/yolov5_logfile.txt"


# -------------------------------
# Resolutions to test
# -------------------------------
resolutions = {
    "original_1920x1080": (1920, 1080),
    "1280x704": (1280, 704),
    "960x512": (960, 512),
}


# -------------------------------
# Run
# -------------------------------
process_images(
    input_directory=input_directory,
    output_root_directory=output_root_directory,
    log_file=log_file,
    resolutions=resolutions,
    conf_thres=0.25,
    iou_thres=0.45,
)