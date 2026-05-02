# artifact_evaluation
# System Requirements

The artifact was tested on Ubuntu Linux with an NVIDIA GPU.

Recommended environment:
- Ubuntu 20.04 or 22.04
- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA 11.x or 12.x
- PyTorch with CUDA enabled
- cuDNN compatible with the installed PyTorch/CUDA version

## Running Steps

This section summarizes the main steps used to reproduce the latency measurement and late-frame detection experiments.

### 1. Organize benign camera images

First, organize the original nuScenes camera images into six camera-view folders:

```text
benign_all/
├── front/
├── front_left/
├── front_right/
├── back/
├── back_left/
└── back_right/

### 2. Run YOLOv5 + SORT on benign data

### 3. Merge benign logs

Merge the generated YOLOv5 + SORT logs so that each timestamp/sample contains one entry from each of the six camera views. The merged output should preserve the six-camera structure used by the late-frame detection method.

### 4. Split benign data for threshold selection and testing
20%  threshold-selection split
80%  benign test split

### 5. Compute detection thresholds

Using the 20% threshold-selection split, compute the benign timing statistics and threshold values used by the late-frame detector. In our experiments, the operating FPS is selected based on the benign P99 end-to-end processing time.


### 6. Prepare nighttime benign and attacked data

Separate the nighttime data into six camera-view folders for both benign and attacked settings:

night_benign/
├── front/
├── front_left/
├── front_right/
├── back/
├── back_left/
└── back_right/

night_attack/
├── front/
├── front_left/
├── front_right/
├── back/
├── back_left/
└── back_right/

### 7. Overlay attack patterns

Overlay the generated attack pattern on the images in the night_attack/ directory. The benign nighttime images remain unchanged.

### 8. Run YOLOv5 + SORT on nighttime benign and attacked data

Run the YOLOv5 + SORT pipeline on both night_benign/ and night_attack/ five times:
Store the logs for benign nighttime runs and attacked nighttime runs separately.

### 9. Merge nighttime logs

Merge the logs from the nighttime benign and attacked runs using the same six-camera format as in Step 3. Each merged sample should contain one entry from each camera view.

### 10. Run the detection method

Run the late-frame detection method on three settings:

1. Benign full test set: daytime + nighttime benign data
2. Benign nighttime data only
3. Attacked nighttime data

The detector uses the merged YOLOv5 + SORT logs to compute late frames per one-second window and evaluate the detection rules.

