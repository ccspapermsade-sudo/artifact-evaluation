import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from optimizationgridimagev2 import (
    set_seed,
    load_tiles,
    load_model,
    lock_M_for_plane,
    eval_camera_design,
    precompensate_projector_gamma,
    H_CONST,
    DEFAULTS,
)


def ensure_yolo_threshold_args(args):
    if getattr(args, "pre_tau", None) is None:
        setattr(args, "pre_tau", DEFAULTS["pre_tau"])
    if getattr(args, "post_tau", None) is None:
        setattr(args, "post_tau", DEFAULTS["post_tau"])
    if getattr(args, "iou_thres", None) is None:
        setattr(args, "iou_thres", DEFAULTS["iou_thres"])
    if getattr(args, "max_det", None) is None:
        setattr(args, "max_det", DEFAULTS["max_det"])
    return args


class BudgetScheduler(BaseCallback):
    """Linearly anneal env.base_budget from start -> final over anneal_steps timesteps."""

    def __init__(self, env, start, final, anneal_steps, verbose=0):
        super().__init__(verbose)
        self.env_ref = env
        self.start = int(start)
        self.final = int(final)
        self.anneal_steps = max(1, int(anneal_steps))

    def _on_training_start(self) -> None:
        self._set_budget(self.start)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        frac = min(1.0, t / self.anneal_steps)
        cur = int(round(self.start + frac * (self.final - self.start)))
        self._set_budget(cur)
        return True

    def _set_budget(self, value: int):
        if hasattr(self.env_ref, "envs"):  # VecEnv
            for e in self.env_ref.envs:
                if hasattr(e, "base_budget"):
                    e.base_budget = value
        else:
            if hasattr(self.env_ref, "base_budget"):
                self.env_ref.base_budget = value


class ExplorationMonitorCallback(BaseCallback):
    """Log PPO exploration state (entropy coefficient and learned log_std) during training."""

    def __init__(self, out_dir, log_every=1000, verbose=0):
        super().__init__(verbose)
        self.out_dir = str(out_dir)
        self.log_every = max(1, int(log_every))
        self.f = None

    def _on_training_start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, "exploration_monitor.txt")
        self.f = open(path, "w", buffering=1)
        self.f.write("# timestep,ent_coef,log_std_mean,log_std_min,log_std_max\n")
        self._write_row()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            self._write_row()
        return True

    def _on_training_end(self) -> None:
        if self.f is not None:
            self._write_row()
            self.f.close()
            self.f = None

    def _write_row(self):
        model = self.model
        ent_coef = float(model.ent_coef) if hasattr(model, "ent_coef") else float("nan")

        if hasattr(model, "policy") and hasattr(model.policy, "log_std"):
            vals = model.policy.log_std.detach().cpu().numpy().reshape(-1)
            log_std_mean = float(np.mean(vals))
            log_std_min = float(np.min(vals))
            log_std_max = float(np.max(vals))
        else:
            log_std_mean = float("nan")
            log_std_min = float("nan")
            log_std_max = float("nan")

        self.f.write(
            f"{self.num_timesteps},{ent_coef:.6f},{log_std_mean:.6f},{log_std_min:.6f},{log_std_max:.6f}\n"
        )


class CameraGridPlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tiles, model, args):
        super().__init__()
        self.tiles = tiles
        self.tile = tiles[0]  # FIXED: always use the first (and only desired) tile
        self.model = model
        self.args = args

        # --- Plane/camera geometry ---
        self.plane_w = int(args.plane_w)
        self.plane_h = int(args.plane_h)

        # M maps PLANE -> CAMERA, with camera canvas outW/outH
        self.M, self.outW, self.outH = lock_M_for_plane(
            H_CONST,
            self.plane_w,
            self.plane_h,
            args.anchor_edge,
            args.max_out_w,
            args.max_out_h,
            pad=0.0,
        )
        self.M_inv = np.linalg.inv(self.M)

        # --- Episode budget ---
        self.base_budget = int(args.budget)
        self.budget = self.base_budget
        self.miss_penalty = float(args.miss_penalty)

        # --- Tile sizing/rotation ---
        self.base_D_list = args.D_list if hasattr(args, "D_list") and len(args.D_list) > 0 else [96, 112, 128, 144, 160]
        self.base_D = min(self.base_D_list)
        self.size_min = float(args.size_min)
        self.size_max = float(args.size_max)
        self.rot_max = float(args.rot_max)

        # --- Spacing (legacy args; now used as packing gaps) ---
        gap_legacy = int(args.grid_gap_px)
        self.gap_x = int(args.grid_gap_x_px) if args.grid_gap_x_px is not None else gap_legacy
        self.gap_y = int(args.grid_gap_y_px) if args.grid_gap_y_px is not None else gap_legacy

        # Near-edge meaning in CAMERA (bottom/top)
        self.near_edge = args.near_edge

        # --- Reward weights ---
        self.w_det = float(args.w_det)
        self.w_total = float(args.w_total)
        self.post_norm = float(args.post_norm)
        self.lambda_local = float(getattr(args, "lambda_local", 0.0))

        self.w_big = float(args.w_big)
        self.size_target = float(args.size_target)
        self.near_big_boost = float(args.near_big_boost)
        self.big_discount_on_success = float(args.big_discount_on_success)

        # Optional distance-aware schedule shaping
        self.w_size_sched = float(getattr(args, "w_size_sched", 0.25))
        self.w_rot_sched = float(getattr(args, "w_rot_sched", 0.05))
        self.size_near_target = float(getattr(args, "size_near_target", 1.0))
        self.size_far_target = float(getattr(args, "size_far_target", 1.4))

        # overlap avoidance in CAMERA space (optional)
        self.avoid_overlap = bool(args.avoid_overlap)
        self.overlap_iou_thres = float(args.overlap_iou_thres)
        self.overlap_penalty = float(getattr(args, "overlap_penalty", 0.5))
        self.max_tries_per_spot = int(getattr(args, "max_tries_per_spot", 6))
        self.max_attempt_factor = int(getattr(args, "max_attempt_factor", 50))
        self.attempt_idx = 0
        self.place_bonus = float(getattr(args, "place_bonus", 0.02))

        # === Packing / object-count shaping ===
        self.w_count = float(getattr(args, "w_count", 0.0))
        self.w_target = float(getattr(args, "w_target", 0.0))
        self.target_count = int(getattr(args, "target_count", 0))
        self.w_target_progress = float(getattr(args, "w_target_progress", 0.0))
        self.target_bonus = float(getattr(args, "target_bonus", 0.0))
        self.target_shortfall_penalty = float(getattr(args, "target_shortfall_penalty", 0.0))
        self.terminate_on_target = bool(getattr(args, "terminate_on_target", False))

        # Optional: size-tier reward (encourage smaller signs, helps packing)
        self.w_size_bins = float(getattr(args, "w_size_bins", 0.0))
        self.size_bin_rewards_raw = str(getattr(args, "size_bin_rewards", "")).strip()
        self._size_bin_rewards = None
        self._size_bin_edges = None

        # Encourage fitting more placements before running out of camera space
        self.w_early_end = float(getattr(args, "w_early_end", 0.0))

        # Near-only hinge penalty for oversizing
        self.w_size_over = float(getattr(args, "w_size_over", 0.0))

        # --- Cursor-based packing state (in CAMERA space) ---
        self.pack_x = 0
        self.pack_y = 0
        self.row_max_h = 0
        self.tries_in_spot = 0

        # Action: [size_factor, rotation, jitter_x, jitter_y] in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation:
        # [post_prev_norm, best_post_norm, post_now_norm, placed_norm, size_norm, rot_norm, step_norm]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32)

        self.obs_post_scale = 300.0

        self.txt_f = None
        self.end_f = None
        self.eval_mode = False
        self.reset(seed=0)

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        denom = area_a + area_b - inter + 1e-9
        return inter / denom

    def _near_alpha_from_camera_y(self, cam_y: float) -> float:
        cam_y = float(np.clip(cam_y, 0.0, float(self.outH - 1)))
        if self.near_edge == "bottom":
            return float(cam_y / max(1.0, float(self.outH - 1)))
        else:
            return float(1.0 - (cam_y / max(1.0, float(self.outH - 1))))

    def _decode_action(self, a):
        a = np.clip(a, 0.0, 1.0)
        size = self.size_min + a[0] * (self.size_max - self.size_min)
        rot = (a[1] * 2.0 - 1.0) * self.rot_max

        jitter_scale = 0.2 * float(self.base_D)
        jx = (a[2] * 2.0 - 1.0) * jitter_scale
        jy = (a[3] * 2.0 - 1.0) * jitter_scale
        return size, rot, jx, jy

    def _size_tier_factor(self, size: float) -> float:
        """Return a factor in [0,1] where 1.0 means *smallest* tier."""
        if self._size_bin_edges is None or self._size_bin_rewards is None or len(self._size_bin_rewards) == 0:
            if self.size_max <= self.size_min + 1e-9:
                return 0.0
            s = float(size)
            return float(np.clip((self.size_max - s) / (self.size_max - self.size_min), 0.0, 1.0))

        s = float(size)
        import bisect

        i = bisect.bisect_right(self._size_bin_edges, s) - 1
        i = max(0, min(len(self._size_bin_rewards) - 1, i))
        tier = float(self._size_bin_rewards[i])
        max_tier = float(max(self._size_bin_rewards))
        if max_tier <= 1e-9:
            return 0.0
        return float(np.clip(tier / max_tier, 0.0, 1.0))

    def _make_tile_variant(self, tile_img, size, rot):
        if tile_img.mode != "RGBA":
            tile_img = tile_img.convert("RGBA")

        target_edge = max(4, int(round(self.base_D * size)))
        tile_resized = tile_img.resize((target_edge, target_edge), Image.BICUBIC)
        if abs(rot) > 1e-3:
            tile_resized = tile_resized.rotate(rot, resample=Image.BICUBIC, expand=True)

        tw, th = tile_resized.size
        return tile_resized, tw, th, target_edge

    def _paste_tile_camera(self, cam_rgba, tile_variant, tlx, tly):
        cam_rgba.alpha_composite(tile_variant, (int(tlx), int(tly)))

        tw, th = tile_variant.size
        x1 = max(0, min(self.outW - 1, int(tlx)))
        y1 = max(0, min(self.outH - 1, int(tly)))
        x2 = max(0, min(self.outW - 1, int(tlx + tw)))
        y2 = max(0, min(self.outH - 1, int(tly + th)))
        return (x1, y1, x2, y2), tw, th

    def _eval_post_on_camera(self, cam_rgb_np):
        m = eval_camera_design(
            self.model,
            cam_rgb_np,
            self.M,
            self.M_inv,
            self.plane_w,
            self.plane_h,
            self.outW,
            self.outH,
            self.args,
            K=self.args.K,
        )
        post = float(m.get("postNMS", 0.0))
        return post, m

    def _build_obs(self, size_norm, rot_norm):
        post_prev_norm = self.post_prev / self.obs_post_scale
        best_post_norm = self.best_post / self.obs_post_scale
        post_now_norm = self.post_now / self.obs_post_scale
        placed_norm = self.placed / max(1.0, float(self.budget))
        step_norm = self.step_idx / float(max(1, int(self.budget)))
        obs = np.array(
            [
                post_prev_norm,
                best_post_norm,
                post_now_norm,
                placed_norm,
                size_norm,
                rot_norm,
                step_norm,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def _open_txt(self):
        if self.txt_f is None:
            os.makedirs(self.args.out_dir, exist_ok=True)
            path = os.path.join(self.args.out_dir, "rl_steps_camera_seq.txt")
            self.txt_f = open(path, "w", buffering=1)
            self.txt_f.write(
                "# episode,step,placed,post_prev,best_post_before,post_now,delta_best_post,delta_local,delta_local_clipped,success,total_bonus,r_delta,r_local,r_miss,r_total,r_prog,r_term,r_tier,r_count,r_sched,r_over,r_rot,r_place,r_early,r_overlap_skip,reward,tlx,tly,size,rot,big_pen,alpha_near,alpha_far,size_desired\n"
            )

    def _open_end_log(self):
        if self.end_f is None:
            os.makedirs(self.args.out_dir, exist_ok=True)
            path = os.path.join(self.args.out_dir, "rl_episode_end_reasons.txt")
            self.end_f = open(path, "w", buffering=1)
            self.end_f.write(
                "# episode,mode,end_reason,step,placed,attempt_idx,budget,pack_x,pack_y,row_max_h,outW,outH,best_post,post_prev,post_now,reached_target,terminated,truncated,extra\n"
            )

    def _log_episode_end(self, reason, terminated, truncated, extra="", reached_target=False):
        if self.end_f is None:
            self._open_end_log()
        mode = "eval" if getattr(self, "eval_mode", False) else "train"
        self.end_f.write(
            f"{self.episode_idx},{mode},{reason},{self.step_idx},{self.placed},{self.attempt_idx},{self.budget},"
            f"{self.pack_x},{self.pack_y},{self.row_max_h},{self.outW},{self.outH},"
            f"{self.best_post:.3f},{self.post_prev:.3f},{self.post_now:.3f},{int(bool(reached_target))},"
            f"{int(bool(terminated))},{int(bool(truncated))},{extra}\n"
        )

    def _episode_image_dir(self):
        mode = "eval" if getattr(self, "eval_mode", False) else "train"
        d = Path(self.args.out_dir) / "episode_camera_patterns" / mode
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_episode_camera(self, end_reason: str):
        save_dir = self._episode_image_dir()
        filename = (
            f"ep{self.episode_idx:05d}_"
            f"step{self.step_idx:04d}_"
            f"placed{self.placed:04d}_"
            f"bestpost{int(round(self.best_post)):04d}_"
            f"{end_reason}_CAMERA.jpg"
        )
        out_path = save_dir / filename
        cam_np = np.array(self.cam_rgba.convert("RGB"))
        Image.fromarray(cam_np).save(str(out_path), quality=95)

    def _finalize_episode(self, reason, terminated, truncated, extra="", reached_target=False):
        self._log_episode_end(
            reason=reason,
            terminated=terminated,
            truncated=truncated,
            extra=extra,
            reached_target=reached_target,
        )
        if getattr(self, "eval_mode", False):
            self._save_episode_camera(reason)

    def _advance_packing_cursor(self, placed_w: int, placed_h: int):
        self.row_max_h = max(self.row_max_h, int(placed_h))

        self.pack_x += int(placed_w) + int(self.gap_x)

        if self.pack_x + int(placed_w) >= self.outW:
            self.pack_x = 0
            self.pack_y += int(self.row_max_h) + int(self.gap_y)
            self.row_max_h = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if getattr(self, "eval_mode", False):
            self.budget = int(self.args.budget_fill_cap)
        else:
            self.budget = int(self.base_budget)

        self.episode_idx = getattr(self, "episode_idx", -1) + 1
        self.cam_rgba = Image.new("RGBA", (self.outW, self.outH), (255, 255, 255, 255))
        self.placed = 0
        self.step_idx = 0
        self.post_prev = 0.0
        self.best_post = 0.0
        self.post_now = 0.0

        self._size_bin_edges = None
        if getattr(self, "_size_bin_rewards", None) is None:
            raw = str(self.size_bin_rewards_raw).strip()
            if raw:
                try:
                    self._size_bin_rewards = [float(x) for x in raw.split(",") if x.strip() != ""]
                except Exception:
                    self._size_bin_rewards = []
            else:
                self._size_bin_rewards = []
        if self._size_bin_rewards:
            import numpy as _np

            n = len(self._size_bin_rewards)
            self._size_bin_edges = _np.linspace(self.size_min, self.size_max, n + 1)

        self.boxes_camera = []

        self.pack_x = 0
        self.pack_y = 0
        self.row_max_h = 0
        self.tries_in_spot = 0
        self.attempt_idx = 0

        return self._build_obs(0.0, 0.0), {}

    def step(self, action):
        size, rot, jx, jy = self._decode_action(action)

        size_norm = (size - self.size_min) / (self.size_max - self.size_min + 1e-9)
        rot_norm = (rot / self.rot_max + 1.0) * 0.5

        self.attempt_idx += 1
        if self.attempt_idx > int(self.budget) * int(self.max_attempt_factor):
            obs = self._build_obs(size_norm, rot_norm)
            self._finalize_episode(
                reason="max_attempts",
                terminated=True,
                truncated=True,
                extra=f"max_attempt_factor={self.max_attempt_factor}",
            )
            return obs, 0.0, True, True, {
                "post_prev": self.post_prev,
                "best_post": self.best_post,
                "post_now": self.post_now,
                "delta_best_post": 0.0,
                "delta_local": 0.0,
                "delta_local_clipped": 0.0,
                "placed": self.placed,
                "max_attempts": True,
            }

        if self.pack_y >= self.outH:
            obs = self._build_obs(size_norm, rot_norm)
            if self.w_early_end > 0.0 and self.budget > 0:
                rem_frac = float(max(0, self.budget - self.step_idx)) / float(self.budget)
                r = -self.w_early_end * rem_frac
            else:
                r = 0.0
            self._finalize_episode(
                reason="out_of_space_precheck",
                terminated=True,
                truncated=False,
                extra=f"w_early_end={self.w_early_end}",
            )
            return obs, float(r), True, False, {
                "post_prev": self.post_prev,
                "best_post": self.best_post,
                "post_now": self.post_now,
                "delta_best_post": 0.0,
                "delta_local": 0.0,
                "delta_local_clipped": 0.0,
                "placed": self.placed,
                "out_of_space": True,
            }

        tile_var, tw, th, _ = self._make_tile_variant(self.tile, size, rot)

        tlx = float(self.pack_x) + float(jx)
        tly = float(self.pack_y) + float(jy)

        tlx = float(np.clip(tlx, -tw * 0.5, self.outW - 1))
        tly = float(np.clip(tly, -th * 0.5, self.outH - 1))

        cand = (
            max(0, int(tlx)),
            max(0, int(tly)),
            min(self.outW - 1, int(tlx + tw)),
            min(self.outH - 1, int(tly + th)),
        )

        cam_cy = float(tly + th * 0.5)
        alpha_near = self._near_alpha_from_camera_y(cam_cy)
        alpha_far = 1.0 - alpha_near

        size_desired = self.size_near_target + alpha_far * (self.size_far_target - self.size_near_target)

        if self.avoid_overlap and self.boxes_camera:
            if any(self._iou(cand, b) >= self.overlap_iou_thres for b in self.boxes_camera):
                self.tries_in_spot += 1

                rot_abs = abs(rot) / max(1e-9, self.rot_max)

                r_delta = 0.0
                r_local = 0.0
                r_miss = 0.0
                r_total = 0.0
                r_prog = 0.0
                r_term = 0.0
                r_tier = 0.0
                r_count = 0.0
                r_sched = 0.0
                r_over = 0.0
                r_rot = -self.w_rot_sched * float(alpha_near) * float(rot_abs)
                r_place = 0.0
                r_early = 0.0
                r_overlap_skip = -self.overlap_penalty
                success = 0.0
                total_bonus = 0.0
                delta_local = 0.0
                delta_local_clipped = 0.0

                reward = r_overlap_skip + r_sched + r_rot

                if self.tries_in_spot >= self.max_tries_per_spot:
                    self.tries_in_spot = 0
                    self.pack_x += int(max(4, self.base_D * self.size_min)) + int(self.gap_x)
                    if self.pack_x >= self.outW:
                        self.pack_x = 0
                        self.pack_y += int(max(4, self.base_D * self.size_min)) + int(self.gap_y)
                        self.row_max_h = 0

                terminated = (self.step_idx >= self.budget) or (self.pack_y >= self.outH)
                obs = self._build_obs(size_norm, rot_norm)

                if self.txt_f is None:
                    self._open_txt()
                self.txt_f.write(
                    f"{self.episode_idx},{self.step_idx},{self.placed},"
                    f"{self.post_prev:.3f},{self.best_post:.3f},{self.post_now:.3f},{0.0:.3f},{delta_local:.3f},{delta_local_clipped:.3f},{success:.1f},{total_bonus:.3f},"
                    f"{r_delta:.3f},{r_local:.3f},{r_miss:.3f},{r_total:.3f},{r_prog:.3f},{r_term:.3f},{r_tier:.3f},{r_count:.3f},"
                    f"{r_sched:.3f},{r_over:.3f},{r_rot:.3f},{r_place:.3f},{r_early:.3f},{r_overlap_skip:.3f},{reward:.3f},"
                    f"{int(tlx)},{int(tly)},{size:.4f},{rot:.2f},"
                    f"{0.0:.4f},{alpha_near:.3f},{alpha_far:.3f},{size_desired:.3f}\n"
                )

                if terminated:
                    overlap_reason = "out_of_space_overlap_skip" if (self.pack_y >= self.outH) else "budget_overlap_skip"
                    self._finalize_episode(
                        reason=overlap_reason,
                        terminated=terminated,
                        truncated=False,
                        extra=(
                            f"tries_in_spot={self.tries_in_spot};max_tries_per_spot={self.max_tries_per_spot};"
                            f"skipped_overlap=1"
                        ),
                    )

                return obs, float(reward), terminated, False, {
                    "post_prev": self.post_prev,
                    "best_post": self.best_post,
                    "post_now": self.post_now,
                    "delta_best_post": 0.0,
                    "delta_local": delta_local,
                    "delta_local_clipped": delta_local_clipped,
                    "placed": self.placed,
                    "skipped_overlap": True,
                    "r_delta": r_delta,
                    "r_local": r_local,
                    "r_miss": r_miss,
                    "r_total": r_total,
                    "r_prog": r_prog,
                    "r_term": r_term,
                    "r_tier": r_tier,
                    "r_count": r_count,
                    "r_sched": r_sched,
                    "r_over": r_over,
                    "r_rot": r_rot,
                    "r_place": r_place,
                    "r_early": r_early,
                    "r_overlap_skip": r_overlap_skip,
                    "reward_total": reward,
                }

        placed_before = self.placed
        self.step_idx += 1

        prev_post_for_log = self.post_prev
        best_post_before = self.best_post

        new_box, placed_w, placed_h = self._paste_tile_camera(self.cam_rgba, tile_var, tlx, tly)
        self.boxes_camera.append(new_box)
        self.placed += 1
        self.tries_in_spot = 0

        # Raw placement-progress reward; gated later by usefulness.
        r_prog_raw = 0.0
        if self.w_target_progress != 0.0:
            r_prog_raw = float(self.w_target_progress) * float(self.placed - placed_before)

        r_tier = 0.0
        if self.w_size_bins != 0.0:
            r_tier = float(self.w_size_bins) * float(self._size_tier_factor(size))

        cam_rgb = np.array(self.cam_rgba.convert("RGB"))
        post_now, _ = self._eval_post_on_camera(cam_rgb)
        self.post_now = post_now

        delta_best_post = max(0.0, post_now - best_post_before)
        delta_local = post_now - prev_post_for_log
        delta_local_clipped = max(0.0, min(delta_local, 1.0))

        success = 1.0 if delta_best_post > 0.0 else 0.0

        if post_now > self.best_post:
            self.best_post = post_now

        big_pen = 0.0
        total_bonus = post_now / max(1.0, self.post_norm)

        r_delta = self.w_det * delta_best_post
        r_local = self.lambda_local * delta_local_clipped
        r_miss = -(self.miss_penalty * (1.0 - success))
        r_total = self.w_total * total_bonus
        r_sched = 0.0

        r_over = 0.0
        if self.w_size_over > 0.0:
            over = max(0.0, float(size - size_desired))
            r_over = -self.w_size_over * float(alpha_near) * float(over**2)

        rot_abs = abs(rot) / max(1e-9, self.rot_max)
        r_rot = -self.w_rot_sched * float(alpha_near) * float(rot_abs)

        # Only reward progress when it improves the running best post-NMS.
        usefulness = success
        r_prog = r_prog_raw * usefulness
        r_place = 0.0
        r_count = 0.0

        r_term = 0.0
        r_early = 0.0
        r_overlap_skip = 0.0

        reward = (
            r_delta
            + r_local
            + r_miss
            + r_total
            + r_prog
            + r_tier
            + r_sched
            + r_over
            + r_rot
            + r_place
            + r_count
        )

        self.post_prev = post_now

        self._advance_packing_cursor(placed_w, placed_h)

        # Terminal target is checked later using best post-NMS, not placed count.
        reached_target = False
        terminated = (self.step_idx >= self.budget) or (self.pack_y >= self.outH) or (self.terminate_on_target and reached_target)
        truncated = False

        if terminated and (self.pack_y >= self.outH) and (self.step_idx < self.budget) and (self.w_early_end > 0.0) and (self.budget > 0):
            rem_frac = float(self.budget - self.step_idx) / float(self.budget)
            r_early = -self.w_early_end * rem_frac
            reward += r_early

        if (terminated or truncated) and (self.target_count > 0):
            # Terminal objective uses the best post-NMS reached in the episode.
            final_metric = float(self.best_post)
            reached_target = (final_metric >= float(self.target_count))

            if reached_target:
                r_term += float(self.target_bonus)
                if self.w_target != 0.0:
                    r_term += float(self.w_target)
            else:
                short = max(0.0, float(self.target_count) - final_metric)
                if short > 0.0:
                    r_term -= float(self.target_shortfall_penalty) * short
                if self.w_target != 0.0 and self.target_count > 0:
                    r_term += float(self.w_target) * min(1.0, final_metric / float(self.target_count))
            reward += r_term

        if terminated or truncated:
            if reached_target and self.terminate_on_target:
                end_reason = "target_reached"
            elif self.pack_y >= self.outH:
                end_reason = "out_of_space"
            elif self.step_idx >= self.budget:
                end_reason = "budget_reached"
            elif truncated:
                end_reason = "truncated"
            else:
                end_reason = "other"
            shortfall = max(0.0, float(self.target_count) - float(self.best_post)) if self.target_count > 0 else 0.0
            self._finalize_episode(
                reason=end_reason,
                terminated=terminated,
                truncated=truncated,
                reached_target=reached_target,
                extra=(
                    f"shortfall_post={shortfall:.3f};best_post={self.best_post:.3f};r_term={r_term:.3f};success={success:.1f};"
                    f"delta_best_post={delta_best_post:.3f};delta_local={delta_local:.3f}"
                ),
            )

        if self.txt_f is None:
            self._open_txt()
        self.txt_f.write(
            f"{self.episode_idx},{self.step_idx},{self.placed},"
            f"{prev_post_for_log:.3f},{best_post_before:.3f},{post_now:.3f},{delta_best_post:.3f},{delta_local:.3f},{delta_local_clipped:.3f},{success:.1f},{total_bonus:.3f},"
            f"{r_delta:.3f},{r_local:.3f},{r_miss:.3f},{r_total:.3f},{r_prog:.3f},{r_term:.3f},{r_tier:.3f},{r_count:.3f},"
            f"{r_sched:.3f},{r_over:.3f},{r_rot:.3f},{r_place:.3f},{r_early:.3f},{r_overlap_skip:.3f},{reward:.3f},"
            f"{int(tlx)},{int(tly)},{size:.4f},{rot:.2f},"
            f"{big_pen:.4f},{alpha_near:.3f},{alpha_far:.3f},{size_desired:.3f}\n"
        )

        return self._build_obs(size_norm, rot_norm), float(reward), terminated, truncated, {
            "post_prev": prev_post_for_log,
            "best_post": best_post_before,
            "post_now": post_now,
            "delta_best_post": delta_best_post,
            "delta_local": delta_local,
            "delta_local_clipped": delta_local_clipped,
            "placed": self.placed,
            "reached_target": reached_target,
            "success": success,
            "total_bonus": total_bonus,
            "r_delta": r_delta,
            "r_local": r_local,
            "r_miss": r_miss,
            "r_total": r_total,
            "r_prog": r_prog,
            "r_term": r_term,
            "r_tier": r_tier,
            "r_count": r_count,
            "r_sched": r_sched,
            "r_over": r_over,
            "r_rot": r_rot,
            "r_place": r_place,
            "r_early": r_early,
            "r_overlap_skip": r_overlap_skip,
            "reward_total": reward,
        }


def run_train(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tiles, _ = load_tiles(args.tiles_dir)
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles found in {args.tiles_dir}")

    model = load_model(args.weights, args.device)
    args = ensure_yolo_threshold_args(args)

    cfg = vars(args).copy()
    cfg["H_CONST"] = H_CONST.tolist()
    with open(Path(args.out_dir) / "config_camera_v5.json", "w") as f:
        json.dump(cfg, f, indent=2)

    env = CameraGridPlacementEnv(tiles, model, args)

    if args.resume_model and os.path.isfile(args.resume_model):
        print(f"Resuming from: {args.resume_model}")
        model_rl = PPO.load(args.resume_model, env=env, device="auto", print_system_info=False)
    else:
        model_rl = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=max(args.budget, 64),
            batch_size=max(64, args.budget),
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    callbacks = []
    callbacks.append(ExplorationMonitorCallback(args.out_dir, log_every=1000))
    if args.checkpoint_every and args.checkpoint_every > 0:
        ckpt_dir = Path(args.out_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(ckpt_dir), name_prefix="ppo_camera_seq"))

    if args.budget_start is not None and args.budget_final is not None and args.budget_anneal_steps is not None:
        print(f"[BudgetScheduler] {args.budget_start} -> {args.budget_final} over {args.budget_anneal_steps} steps")
        env.base_budget = int(args.budget_start)
        callbacks.append(BudgetScheduler(env, args.budget_start, args.budget_final, args.budget_anneal_steps))

    model_rl.learn(total_timesteps=args.total_timesteps, callback=callbacks if callbacks else None, reset_num_timesteps=False)

    model_path = Path(args.out_dir) / "ppo_camera_pattern_seq.zip"
    model_rl.save(model_path)

    export_dir = Path(args.out_dir) / "rl_camera_top"
    export_dir.mkdir(parents=True, exist_ok=True)

    env.eval_mode = True

    for r in range(1, args.export_rollouts + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model_rl.predict(obs, deterministic=True)

            action = np.array(action, dtype=np.float32).reshape(1, -1)
            if args.export_clamp_rot:
                action[0, 1] = 0.5
            if args.export_clamp_jitter:
                action[0, 2] = 0.5
                action[0, 3] = 0.5

            obs, reward, term, trunc, info = env.step(action[0])
            done = (term or trunc)

        cam_np = np.array(env.cam_rgba.convert("RGB"))

        plane_np = cv2.warpPerspective(
            cam_np,
            env.M_inv,
            (env.plane_w, env.plane_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if args.precompensate_gamma > 0:
            plane_save = precompensate_projector_gamma(plane_np, proj_gamma=args.precompensate_gamma)
        else:
            plane_save = plane_np

        Image.fromarray(cam_np).save(export_dir / f"roll{r:02d}_CAMERA.jpg", quality=95)
        Image.fromarray(plane_save).save(export_dir / f"roll{r:02d}_PLANE.jpg", quality=95)

    print(f"Saved trained policy to: {model_path}")
    print(f"Saved rollout images to: {export_dir}")
    print(f"Saved per-episode CAMERA images to: {Path(args.out_dir) / 'episode_camera_patterns'}")
    

def run_eval(args):
    """
    Evaluation-only mode.

    This does not train or update the PPO policy. It loads a saved PPO model
    from --resume_model, runs deterministic rollouts, and saves the same CAMERA
    and PLANE images as the post-training export in run_train().
    """
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if not args.resume_model or not os.path.isfile(args.resume_model):
        raise RuntimeError("For --mode eval, please provide a valid --resume_model path to a .zip PPO model.")

    tiles, _ = load_tiles(args.tiles_dir)
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles found in {args.tiles_dir}")

    model = load_model(args.weights, args.device)
    args = ensure_yolo_threshold_args(args)

    cfg = vars(args).copy()
    cfg["H_CONST"] = H_CONST.tolist()
    with open(Path(args.out_dir) / "config_camera_v5_eval.json", "w") as f:
        json.dump(cfg, f, indent=2)

    env = CameraGridPlacementEnv(tiles, model, args)
    env.eval_mode = True

    print(f"Loading trained policy for evaluation from: {args.resume_model}")
    model_rl = PPO.load(args.resume_model, env=env, device="auto", print_system_info=False)

    export_dir = Path(args.out_dir) / "rl_camera_eval"
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.out_dir) / "eval_summary.csv"
    with open(summary_path, "w") as f:
        f.write("rollout,total_reward,best_post,final_post,placed,steps,terminated,truncated\n")

        for r in range(1, args.export_rollouts + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            last_info = {}
            term = False
            trunc = False

            while not done:
                action, _ = model_rl.predict(obs, deterministic=True)

                action = np.array(action, dtype=np.float32).reshape(1, -1)
                if args.export_clamp_rot:
                    action[0, 1] = 0.5
                if args.export_clamp_jitter:
                    action[0, 2] = 0.5
                    action[0, 3] = 0.5

                obs, reward, term, trunc, info = env.step(action[0])
                total_reward += float(reward)
                last_info = info
                done = (term or trunc)

            cam_np = np.array(env.cam_rgba.convert("RGB"))

            plane_np = cv2.warpPerspective(
                cam_np,
                env.M_inv,
                (env.plane_w, env.plane_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            if args.precompensate_gamma > 0:
                plane_save = precompensate_projector_gamma(plane_np, proj_gamma=args.precompensate_gamma)
            else:
                plane_save = plane_np

            Image.fromarray(cam_np).save(export_dir / f"eval_roll{r:02d}_CAMERA.jpg", quality=95)
            Image.fromarray(plane_save).save(export_dir / f"eval_roll{r:02d}_PLANE.jpg", quality=95)

            final_post = float(last_info.get("post_now", getattr(env, "post_now", 0.0)))
            f.write(
                f"{r},{total_reward:.6f},{float(env.best_post):.6f},{final_post:.6f},"
                f"{int(env.placed)},{int(env.step_idx)},{int(bool(term))},{int(bool(trunc))}\n"
            )

            print(
                f"[EVAL] rollout={r}/{args.export_rollouts} "
                f"best_post={env.best_post:.1f} final_post={final_post:.1f} "
                f"placed={env.placed} steps={env.step_idx}"
            )

    print(f"Saved evaluation images to: {export_dir}")
    print(f"Saved evaluation summary to: {summary_path}")
    print(f"Saved per-episode CAMERA images to: {Path(args.out_dir) / 'episode_camera_patterns'}")


def parse_args():
    p = argparse.ArgumentParser()

    # Mode: default keeps the original training behavior unchanged
    p.add_argument("--mode", type=str, choices=["train", "eval"], default="train")

    # YOLO thresholds
    p.add_argument("--pre_tau", type=float, default=0.25)
    p.add_argument("--post_tau", type=float, default=0.45)
    p.add_argument("--iou_thres", type=float, default=0.50)
    p.add_argument("--max_det", type=int, default=1000)

    # Required
    p.add_argument("--tiles_dir", type=str, required=True)
    p.add_argument("--weights", type=str, default="yolov5s.pt")
    p.add_argument("--device", type=str, default="")

    # Geometry
    p.add_argument("--plane_w", type=int, default=1920)
    p.add_argument("--plane_h", type=int, default=1080)
    p.add_argument("--anchor_edge", type=str, choices=["bottom", "left"], default="bottom")
    p.add_argument("--max_out_w", type=int, default=1920)
    p.add_argument("--max_out_h", type=int, default=1080)

    # Placement ranges (interpreted in CAMERA pixels now)
    p.add_argument("--D_list", type=int, nargs="+", default=[96, 112, 128, 144, 160])
    p.add_argument("--size_min", type=float, default=0.5)
    p.add_argument("--size_max", type=float, default=1.5)
    p.add_argument("--rot_max", type=float, default=15.0)

    # Spacing (used as packing gaps)
    p.add_argument("--grid_gap_px", type=int, default=0, help="legacy: extra pixels between placements (x and y)")
    p.add_argument("--grid_gap_x_px", type=int, default=None)
    p.add_argument("--grid_gap_y_px", type=int, default=None)

    p.add_argument("--near_edge", type=str, choices=["bottom", "top"], default="bottom")

    # RL
    p.add_argument("--budget", type=int, default=80)
    p.add_argument("--miss_penalty", type=float, default=10.0)
    p.add_argument("--total_timesteps", type=int, default=30000)
    p.add_argument("--export_rollouts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=1)

    # Reward
    p.add_argument("--w_det", type=float, default=1.0)
    p.add_argument("--w_total", type=float, default=0.6)
    p.add_argument("--post_norm", type=float, default=1000.0)
    p.add_argument(
        "--lambda_local",
        type=float,
        default=0.0,
        help="Weight for clipped local reward: max(0, min(post_now - post_prev, 1))",
    )

    p.add_argument("--w_big", type=float, default=0.4)
    p.add_argument("--size_target", type=float, default=1.0)
    p.add_argument("--near_big_boost", type=float, default=2.0)
    p.add_argument("--big_discount_on_success", type=float, default=0.9)

    # Schedule shaping
    p.add_argument("--w_size_sched", type=float, default=0.25)
    p.add_argument("--w_rot_sched", type=float, default=0.05)
    p.add_argument("--size_near_target", type=float, default=1.0)
    p.add_argument("--size_far_target", type=float, default=1.4)

    p.add_argument(
        "--w_early_end",
        type=float,
        default=0.0,
        help="Extra penalty if the episode ends due to running out of camera space before reaching budget",
    )

    p.add_argument(
        "--w_size_over",
        type=float,
        default=0.0,
        help="Additional near-only penalty for (size - size_desired)+^2 to avoid starting with max size near",
    )

    p.add_argument("--overlap_penalty", type=float, default=0.5)
    p.add_argument("--max_tries_per_spot", type=int, default=6)
    p.add_argument(
        "--max_attempt_factor",
        type=int,
        default=50,
        help="Hard cap on total step() calls per episode = budget * max_attempt_factor",
    )
    p.add_argument("--place_bonus", type=float, default=0.02)

    # Overlap + misc
    p.add_argument("--avoid_overlap", action="store_true")
    p.add_argument("--overlap_iou_thres", type=float, default=0.30)
    p.add_argument("--precompensate_gamma", type=float, default=0.0)
    p.add_argument("--out_dir", type=str, default="runs/pattern_rl_camera_seq")

    # Resume/checkpoints/curriculum
    p.add_argument("--resume_model", type=str, default="")
    p.add_argument("--checkpoint_every", type=int, default=0)
    p.add_argument("--budget_fill_cap", type=int, default=9999)

    p.add_argument("--budget_start", type=int, default=None)
    p.add_argument("--budget_final", type=int, default=None)
    p.add_argument("--budget_anneal_steps", type=int, default=None)

    # Export clamping
    p.add_argument("--export_clamp_rot", action="store_true")
    p.add_argument("--export_clamp_jitter", action="store_true")

    # Packing / object-count shaping
    p.add_argument("--w_count", type=float, default=0.0, help="Extra reward per successful placement.")
    p.add_argument("--target_count", type=int, default=0, help="If >0: aim to place this many objects.")
    p.add_argument("--terminate_on_target", action="store_true")
    p.add_argument("--target_bonus", type=float, default=50.0)
    p.add_argument("--target_shortfall_penalty", type=float, default=0.25)
    p.add_argument("--w_target_progress", type=float, default=0.0)
    p.add_argument("--w_target", type=float, default=0.0)

    # Optional: discrete size-tier reward (smaller -> higher by default)
    p.add_argument("--size_bin_rewards", type=str, default="", help="Comma-separated rewards for size tiers (small->large).")
    p.add_argument("--w_size_bins", type=float, default=0.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args = ensure_yolo_threshold_args(args)

    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)
