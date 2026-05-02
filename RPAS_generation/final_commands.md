# Final Commands

This file lists the main commands used to run the artifact experiments.  
Before running the commands, update the paths below according to your local environment.

```bash
# Root paths
export PROJECT_DIR=/path/to/artifact-evaluation
export YOLOV5_DIR=/path/to/yolov5
export SORT_DIR=/path/to/sort
export TILES_DIR=/path/to/stop_sign_samples
export WEIGHTS=yolov5s.pt
export DEVICE=0

# Optional output root
export OUT_ROOT=/path/to/outputs
```

---

## 1. Run YOLOv5 + SORT

```bash
PYTHONPATH=${SORT_DIR}:$PYTHONPATH \
python dosonbatchcustumresolutiondttracking.py
```

---

## 2. RL Pattern Generation: Full Reward

```bash
python ${YOLOV5_DIR}/train_and_eval.py \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/rl_full_reward \
  --plane_w 1920 \
  --plane_h 1080 \
  --max_out_w 1920 \
  --max_out_h 1080 \
  --target_count 300 \
  --w_target 1.0 \
  --w_target_progress 0 \
  --target_bonus 20 \
  --target_shortfall_penalty 0.18 \
  --budget 320 \
  --budget_fill_cap 420 \
  --total_timesteps 145000 \
  --export_rollouts 5 \
  --checkpoint_every 5000 \
  --miss_penalty 0.5 \
  --post_norm 300 \
  --w_total 0.5 \
  --lambda_local 0.3 \
  --w_size_over 0.25 \
  --w_size_bins 0.15 \
  --grid_gap_x_px -3 \
  --grid_gap_y_px -3 \
  --w_rot_sched 0 \
  --size_bin_rewards "1.0,0.6,0.3,0.2"
```

---

## 3. RL Evaluation

Use the trained checkpoint from the full-reward run and run evaluation mode.

```bash
python ${YOLOV5_DIR}/train_and_eval.py \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/rl_full_reward_eval \
  --plane_w 1920 \
  --plane_h 1080 \
  --max_out_w 1920 \
  --max_out_h 1080 \
  --target_count 300 \
  --w_target 1.0 \
  --w_target_progress 0 \
  --target_bonus 20 \
  --target_shortfall_penalty 0.18 \
  --budget 320 \
  --budget_fill_cap 420 \
  --export_rollouts 5 \
  --miss_penalty 0.5 \
  --post_norm 300 \
  --w_total 0.5 \
  --lambda_local 0.3 \
  --w_size_over 0.25 \
  --w_size_bins 0.15 \
  --grid_gap_x_px -3 \
  --grid_gap_y_px -3 \
  --w_rot_sched 0 \
  --size_bin_rewards "1.0,0.6,0.3,0.2" \
  --eval_mode
```

If your evaluation script requires a checkpoint argument, use the trained checkpoint explicitly, for example:

```bash
python ${YOLOV5_DIR}/train_and_eval.py \
  --eval_mode \
  --checkpoint ${OUT_ROOT}/rl_full_reward/checkpoints/best_model.zip \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/rl_full_reward_eval
```

---

## 4. Reward Ablation Commands

The following ablations use the same full-reward command above, with only the listed option changed.

### 4.1 No detection-gain reward, no \(r_t^\Delta\)

```bash
# Add this option to the full-reward command:
--w_det 0
```

### 4.2 No local reward, no \(r_t^{local}\)

```bash
# Replace the default local-reward weight:
--lambda_local 0
```

### 4.3 No miss penalty, no \(r_t^{miss}\)

```bash
# Replace the default miss penalty:
--miss_penalty 0
```

### 4.4 No total-count reward, no \(r_t^{total}\)

```bash
# Replace the default total-count reward weight:
--w_total 0
```

### 4.5 No density shaping, no \(r_t^{dens}\)

```bash
# Replace the default density-shaping parameters:
--target_bonus 0 --target_shortfall_penalty 0 --w_target 0
```

### 4.6 No size reward, no \(r_t^{size}\)

```bash
# Replace the default size-bin reward weight:
--w_size_bins 0
```

### 4.7 No near-oversize penalty, no \(r_t^{near\text{-}oversize}\)

```bash
# Replace the default near-oversize penalty weight:
--w_size_over 0
```

---

## 5. Baseline Comparison

All baselines use the same detector, camera-plane geometry, placement budget, and evaluation mode.

### 5.1 Heuristic Baseline

```bash
python ${YOLOV5_DIR}/heuristic_baseline_camera_eval.py \
  --rl_py ${YOLOV5_DIR}/train_and_eval.py \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/heuristic_baseline \
  --plane_w 1920 \
  --plane_h 1080 \
  --max_out_w 1920 \
  --max_out_h 1080 \
  --target_count 300 \
  --w_target 1.0 \
  --w_target_progress 0 \
  --target_bonus 20 \
  --target_shortfall_penalty 0.18 \
  --budget 320 \
  --budget_fill_cap 420 \
  --w_rot_sched 0 \
  --miss_penalty 0.5 \
  --post_norm 300 \
  --w_total 0.5 \
  --lambda_local 0.3 \
  --w_size_over 0.25 \
  --w_size_bins 0.15 \
  --size_bin_rewards "1.0,0.6,0.3,0.2" \
  --episodes 100 \
  --seed 0 \
  --eval_mode \
  --size_action 0.30 \
  --rot_action 0.50 \
  --jx_action 0.50 \
  --jy_action 0.50
```

### 5.2 Random-Search-with-Budget Baseline

```bash
python ${YOLOV5_DIR}/random_search_with_budget_camera_eval.py \
  --rl_py ${YOLOV5_DIR}/train_and_eval.py \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/random_search_baseline \
  --plane_w 1920 \
  --plane_h 1080 \
  --max_out_w 1920 \
  --max_out_h 1080 \
  --target_count 300 \
  --w_target 1.0 \
  --w_target_progress 0 \
  --target_bonus 20 \
  --target_shortfall_penalty 0.18 \
  --budget 320 \
  --budget_fill_cap 420 \
  --w_rot_sched 0 \
  --miss_penalty 0.5 \
  --post_norm 300 \
  --w_total 0.5 \
  --lambda_local 0.3 \
  --w_size_over 0.25 \
  --w_size_bins 0.15 \
  --size_bin_rewards "1.0,0.6,0.3,0.2" \
  --episodes 100 \
  --seed 0 \
  --eval_mode
```

### 5.3 Random Agent Baseline

```bash
python ${YOLOV5_DIR}/random_baseline_camera_eval.py \
  --rl_py ${YOLOV5_DIR}/train_and_eval.py \
  --tiles_dir ${TILES_DIR} \
  --weights ${WEIGHTS} \
  --device ${DEVICE} \
  --out_dir ${OUT_ROOT}/random_agent_baseline \
  --plane_w 1920 \
  --plane_h 1080 \
  --max_out_w 1920 \
  --max_out_h 1080 \
  --target_count 300 \
  --w_target 1.0 \
  --w_target_progress 0 \
  --target_bonus 20 \
  --target_shortfall_penalty 0.18 \
  --budget 320 \
  --budget_fill_cap 420 \
  --w_rot_sched 0 \
  --miss_penalty 0.5 \
  --post_norm 300 \
  --w_total 0.5 \
  --lambda_local 0.3 \
  --w_size_over 0.25 \
  --w_size_bins 0.15 \
  --size_bin_rewards "1.0,0.6,0.3,0.2" \
  --episodes 100 \
  --seed 0 \
  --eval_mode
```

---

## Notes

- Replace all placeholder paths, such as `/path/to/yolov5`, with the correct local paths before running.
