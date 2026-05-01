import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from sort import Sort


# ============================================================
# Configuration
# ============================================================
root_dir = "/home..."
#custom_resolution = (2592, 4608)  # (width, height)
custom_resolution = (1088,1920)
#custom_resolution = (512,960)
#custom_resolution = (704,1280)
#custom_resolution = (896,1600)
#custom_resolution = (832,1440)
#custom_resolution = (768,1312)
#custom_resolution = (640,640)
# custom_resolution = (2944, 3968)

conf_thres = 0.25
iou_thres = 0.45
max_det = 10000

weights = "yolov5s.pt"  # or 'yolov5x6.pt'


# ============================================================
# Helper: run SORT update and time it
# ============================================================
def run_sort_update(tracker: Sort, pred0):
    """
    pred0: YOLOv5 NMS output for one image, tensor Nx6 [x1,y1,x2,y2,conf,cls]
    Returns: (track_ms, num_tracks)
    """
    if pred0 is None or len(pred0) == 0:
        dets_sort = np.empty((0, 5), dtype=np.float32)
    else:
        # SORT expects Nx5 [x1,y1,x2,y2,score]
        dets_sort = pred0[:, :5].detach().cpu().numpy().astype(np.float32)

    t0 = time.perf_counter()
    tracks = tracker.update(dets_sort)  # returns Nx5 [x1,y1,x2,y2,track_id]
    t1 = time.perf_counter()

    track_ms = (t1 - t0) * 1000.0
    num_tracks = 0 if tracks is None else tracks.shape[0]
    return track_ms, num_tracks


# ============================================================
# Initialize device and model
# ============================================================
device = select_device("0")  # '' lets YOLOv5 choose (GPU if available)
print(f"Using device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Warmup (H, W order used here is imgsz=(1,3,H,W))
model.warmup(imgsz=(1, 3, custom_resolution[1], custom_resolution[0]))


# ============================================================
# Process batch directories
# ============================================================
batch_dirs = [
    os.path.join(root_dir, d)
    for d in sorted(os.listdir(root_dir))
    if os.path.isdir(os.path.join(root_dir, d))
]

for batch_dir in batch_dirs:
    batch_name = os.path.basename(batch_dir)
    output_file = os.path.join(root_dir, f"{batch_name}.txt")

    # IMPORTANT: init tracker per batch/sequence
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

    image_files = sorted([f for f in os.listdir(batch_dir) if f.lower().endswith((".jpg", ".png"))])
    log_data = []
    total_start_time = time.time()

    for image_file in image_files:
        image_path = os.path.join(batch_dir, image_file)
        img0 = cv2.imread(image_path)
        assert img0 is not None, f"Image {image_path} not found"

        orig_height, orig_width = img0.shape[:2]
        image_start_time = time.time()

        # Preprocessing 
        pre_start = time.time()
        img = letterbox(img0, new_shape=custom_resolution, auto=False)[0]
        resized_height, resized_width = img.shape[:2]
        img = img.transpose((2, 0, 1))[::-1]  # HWC->CHW and BGR->RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float() / 255.0
        img = img.unsqueeze(0)
        pre_end = time.time()

        # Inference 
        torch.cuda.synchronize()
        t1 = time.time()
        pred_tensor = model(img)[0]  # (1, num_preds, 85)
        torch.cuda.synchronize()
        t2 = time.time()

        # Extract pre-NMS predictions
        raw_pred = pred_tensor[0]  # (num_preds, 85)
        num_boxes_before_nms = raw_pred.shape[0]

        with torch.no_grad():
            obj_scores = raw_pred[:, 4].detach().cpu()
            num_boxes_above_thresh = (obj_scores > conf_thres).sum().item()

        # NMS
        t_nms_start = time.time()
        pred = non_max_suppression(
            pred_tensor,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            agnostic=True,
            max_det=max_det,
        )
        t3 = time.time()

        num_boxes_after_nms = len(pred[0]) if pred[0] is not None else 0

        # SORT Tracking 
        track_ms, num_tracks = run_sort_update(tracker, pred[0])

        total_image_time = time.time() - image_start_time

        # Log Info 
        log_data.append(
            f"{image_file}: Original {orig_width}x{orig_height}, Model Input {resized_width}x{resized_height}, "
            f"Preprocess {(pre_end - pre_start)*1000:.2f}ms, "
            f"Inference {(t2 - t1)*1000:.2f}ms, "
            f"Postprocess(NMS) {(t3 - t_nms_start)*1000:.2f}ms, "
            f"Tracking(SORT) {track_ms:.2f}ms, "
            f"Total {total_image_time:.2f}s, "
            f"Before NMS (anchors): {num_boxes_before_nms}, "
            f"Above Conf Thresh: {num_boxes_above_thresh}, "
            f"After NMS: {num_boxes_after_nms}, "
            f"Tracks: {num_tracks}"
        )

    # Summary for batch
    total_elapsed_time = time.time() - total_start_time
    log_data.append(f"\n Total Processing Time: {total_elapsed_time:.2f} seconds")
    log_data.append(f" Processed {len(image_files)} images in batch {batch_name}.")

    # Save log to file
    with open(output_file, "w") as f:
        f.write("\n".join(log_data) + "\n")

    print(f" Processed {len(image_files)} images in batch {batch_name} in {total_elapsed_time:.2f} seconds")
    print(f" Log saved to {output_file}")
