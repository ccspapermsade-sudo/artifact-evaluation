#!/usr/bin/env python3
"""
Random-search-with-budget baseline for the camera-first RL environment.

What it does:
- loads the original RL environment from --rl_py
- runs many random episodes
- samples actions uniformly from [0,1]^4 at each step
- keeps the best final design found across all episodes
- saves:
    * per-episode CSV
    * summary JSON
    * best design images (CAMERA / PLANE)
    * optional rollout images for all episodes
"""

import os
import sys
import csv
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


def load_rl_module(py_path: str):
    py_path = os.path.abspath(py_path)
    rl_dir = os.path.dirname(py_path)

    if rl_dir in sys.path:
        sys.path.remove(rl_dir)
    sys.path.insert(0, rl_dir)

    spec = importlib.util.spec_from_file_location("rl_train_module", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--rl_py", type=str, required=True,
                   help="Path to the RL training file")
    p.add_argument("--tiles_dir", type=str, required=True)
    p.add_argument("--weights", type=str, default="yolov5s.pt")
    p.add_argument("--device", type=str, default="")

    p.add_argument("--plane_w", type=int, default=1920)
    p.add_argument("--plane_h", type=int, default=1080)
    p.add_argument("--anchor_edge", type=str, choices=["bottom", "left"], default="bottom")
    p.add_argument("--max_out_w", type=int, default=1920)
    p.add_argument("--max_out_h", type=int, default=1080)

    p.add_argument("--D_list", type=int, nargs="+", default=[96, 112, 128, 144, 160])
    p.add_argument("--size_min", type=float, default=0.5)
    p.add_argument("--size_max", type=float, default=1.5)
    p.add_argument("--rot_max", type=float, default=15.0)

    p.add_argument("--grid_gap_px", type=int, default=0)
    p.add_argument("--grid_gap_x_px", type=int, default=None)
    p.add_argument("--grid_gap_y_px", type=int, default=None)

    p.add_argument("--near_edge", type=str, choices=["bottom", "top"], default="bottom")

    p.add_argument("--budget", type=int, default=80)
    p.add_argument("--budget_fill_cap", type=int, default=9999)
    p.add_argument("--miss_penalty", type=float, default=10.0)
    p.add_argument("--K", type=int, default=1)

    p.add_argument("--pre_tau", type=float, default=0.25)
    p.add_argument("--post_tau", type=float, default=0.45)
    p.add_argument("--iou_thres", type=float, default=0.50)
    p.add_argument("--max_det", type=int, default=1000)

    p.add_argument("--w_det", type=float, default=1.0)
    p.add_argument("--w_total", type=float, default=0.6)
    p.add_argument("--post_norm", type=float, default=1000.0)
    p.add_argument("--lambda_local", type=float, default=0.0)

    p.add_argument("--w_big", type=float, default=0.4)
    p.add_argument("--size_target", type=float, default=1.0)
    p.add_argument("--near_big_boost", type=float, default=2.0)
    p.add_argument("--big_discount_on_success", type=float, default=0.9)

    p.add_argument("--w_size_sched", type=float, default=0.25)
    p.add_argument("--w_rot_sched", type=float, default=0.05)
    p.add_argument("--size_near_target", type=float, default=1.0)
    p.add_argument("--size_far_target", type=float, default=1.4)

    p.add_argument("--w_early_end", type=float, default=0.0)
    p.add_argument("--w_size_over", type=float, default=0.0)

    p.add_argument("--overlap_penalty", type=float, default=0.5)
    p.add_argument("--max_tries_per_spot", type=int, default=6)
    p.add_argument("--max_attempt_factor", type=int, default=50)
    p.add_argument("--place_bonus", type=float, default=0.02)

    p.add_argument("--avoid_overlap", action="store_true")
    p.add_argument("--overlap_iou_thres", type=float, default=0.30)
    p.add_argument("--precompensate_gamma", type=float, default=0.0)

    p.add_argument("--w_count", type=float, default=0.0)
    p.add_argument("--target_count", type=int, default=0)
    p.add_argument("--terminate_on_target", action="store_true")
    p.add_argument("--target_bonus", type=float, default=50.0)
    p.add_argument("--target_shortfall_penalty", type=float, default=0.25)
    p.add_argument("--w_target_progress", type=float, default=0.0)
    p.add_argument("--w_target", type=float, default=0.0)

    p.add_argument("--size_bin_rewards", type=str, default="")
    p.add_argument("--w_size_bins", type=float, default=0.0)

    p.add_argument("--episodes", type=int, default=100,
                   help="Number of random episodes in the search budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_mode", action="store_true",
                   help="Use budget_fill_cap like exported RL rollouts")
    p.add_argument("--save_every", type=int, default=0,
                   help="Save all rollout images every N episodes; 0 disables")
    p.add_argument("--out_dir", type=str, required=True)

    return p.parse_args()


def save_rollout_images(env, save_dir: Path, prefix: str):
    cam_np = np.array(env.cam_rgba.convert("RGB"))

    plane_np = cv2.warpPerspective(
        cam_np,
        env.M_inv,
        (env.plane_w, env.plane_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    Image.fromarray(cam_np).save(save_dir / f"{prefix}_CAMERA.jpg", quality=95)
    Image.fromarray(plane_np).save(save_dir / f"{prefix}_PLANE.jpg", quality=95)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_mod = load_rl_module(args.rl_py)
    args = rl_mod.ensure_yolo_threshold_args(args)
    rl_mod.set_seed(args.seed)

    tiles, _ = rl_mod.load_tiles(args.tiles_dir)
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles found in {args.tiles_dir}")

    model = rl_mod.load_model(args.weights, args.device)
    env = rl_mod.CameraGridPlacementEnv(tiles, model, args)
    env.eval_mode = bool(args.eval_mode)

    all_rollouts_dir = out_dir / "random_search_rollouts"
    all_rollouts_dir.mkdir(parents=True, exist_ok=True)

    best_dir = out_dir / "best_design"
    best_dir.mkdir(parents=True, exist_ok=True)

    cfg = vars(args).copy()
    with open(out_dir / "random_search_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    csv_path = out_dir / "random_search_results.csv"
    rows = []

    rng = np.random.default_rng(args.seed)

    best_score = -np.inf
    best_episode = None
    best_row = None

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        last_info = {}

        while not done:
            action = rng.uniform(0.0, 1.0, size=(4,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            last_info = info

        row = {
            "episode": ep,
            "reward_sum": float(ep_reward),
            "best_post_final": float(env.best_post),
            "post_now_final": float(env.post_now),
            "post_prev_final": float(env.post_prev),
            "placed_final": int(env.placed),
            "step_idx_final": int(env.step_idx),
            "budget_used": int(env.budget),
            "reached_target": int(bool(last_info.get("reached_target", False))),
            "out_of_space": int(bool(last_info.get("out_of_space", False))),
            "max_attempts": int(bool(last_info.get("max_attempts", False))),
        }
        rows.append(row)

        print(
            f"[Random-search ep {ep:03d}] "
            f"reward_sum={row['reward_sum']:.3f}  "
            f"best_post={row['best_post_final']:.3f}  "
            f"post_now={row['post_now_final']:.3f}  "
            f"placed={row['placed_final']}  "
            f"budget={row['budget_used']}"
        )

        if args.save_every > 0 and (ep % args.save_every == 0):
            save_rollout_images(env, all_rollouts_dir, f"ep{ep:03d}")

        score = row["best_post_final"]
        if score > best_score:
            best_score = score
            best_episode = ep
            best_row = row.copy()

            for p in best_dir.glob("*"):
                if p.is_file():
                    p.unlink()

            save_rollout_images(env, best_dir, "best")
            with open(best_dir / "best_episode_info.json", "w") as f:
                json.dump(best_row, f, indent=2)

    if len(rows) == 0:
        raise RuntimeError("No episodes were run.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_posts = np.array([r["best_post_final"] for r in rows], dtype=np.float32)
    rewards = np.array([r["reward_sum"] for r in rows], dtype=np.float32)
    placeds = np.array([r["placed_final"] for r in rows], dtype=np.float32)

    summary = {
        "episodes": int(args.episodes),
        "search_budget_episodes": int(args.episodes),
        "best_post_mean": float(best_posts.mean()),
        "best_post_std": float(best_posts.std()),
        "best_post_median": float(np.median(best_posts)),
        "best_post_max": float(best_posts.max()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "placed_mean": float(placeds.mean()),
        "success_ge_50": float(np.mean(best_posts >= 50.0)),
        "success_ge_100": float(np.mean(best_posts >= 100.0)),
        "success_ge_200": float(np.mean(best_posts >= 200.0)),
        "success_ge_300": float(np.mean(best_posts >= 300.0)),
        "best_episode": int(best_episode),
        "best_design_best_post": float(best_row["best_post_final"]),
        "best_design_post_now": float(best_row["post_now_final"]),
        "best_design_reward_sum": float(best_row["reward_sum"]),
        "best_design_placed": int(best_row["placed_final"]),
    }

    with open(out_dir / "random_search_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Random-search-with-budget summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-episode CSV to: {csv_path}")
    print(f"Saved best design to: {best_dir}")


if __name__ == "__main__":
    main()
