#!/usr/bin/env python3
# coding: utf-8 
"""
Modes:
  --mode plane   : optimize tile layout on the PLANE; score in CAMERA (Phase A + Phase B).
  --mode camera  : optimize tile layout in the CAMERA image (tilted ~20°); back-map to PLANE with M^-1,
                   then forward again to score (Phase A + Phase B with relaxed gates).
"""

import os, sys, time, json, math, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ---- YOLOv5 (run from repo root or add to PYTHONPATH) ----
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# ---- OpenCV ----
import cv2

# =========================
# HARD-CODED HOMOGRAPHY (plane -> camera, normalized)
# =========================
H_CONST = np.array([
    [-1.116334522350972,  0.732016051296136,  1.453217882544693e+03],
    [ 0.045731942734363, -0.095353747805466,  9.711108944476850e+02],
    [ 2.398879009732464e-05, 7.448285936527822e-04, 1.0]
], dtype=np.float64)
H_CONST /= H_CONST[2, 2]

# =========================
# Defaults (YOLO thresholds unchanged)
# =========================
DEFAULTS = dict(
    pre_tau=0.10,
    post_tau=0.25,
    iou_thres=0.45,
    max_det=30000,
    imgsz=(1080, 1920),
    score_weights=dict(preNMS=1.0, postNMS=0.2, nms_ms=0.9),
    save_top_k=12,
)

# ----------------- utils -----------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)

def load_tiles(tile_dir):
    paths = sorted([str(p) for p in Path(tile_dir).glob("*.png")])
    if not paths:
        raise FileNotFoundError(f"No PNG tiles found in {tile_dir}")
    return [Image.open(p).convert("RGBA") for p in paths], paths

def apply_gamma(img_rgb, gamma):
    x = img_rgb.astype(np.float32) / 255.0
    y = np.clip(x ** gamma, 0, 1)
    return (y * 255).astype(np.uint8)

def apply_wb(img_rgb, r=1.0, g=1.0, b=1.0):
    x = img_rgb.astype(np.float32)
    x[...,0] *= r; x[...,1] *= g; x[...,2] *= b
    return np.clip(x, 0, 255).astype(np.uint8)

def add_noise(img_rgb, sigma=3.0):
    noise = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
    x = img_rgb.astype(np.float32) + noise
    return np.clip(x, 0, 255).astype(np.uint8)

def gaussian_blur(img_rgb, sigma=1.0):
    k = int(max(3, 2*int(3*sigma)+1))
    return cv2.GaussianBlur(img_rgb, (k,k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

def small_perspective(img_rgb, max_ratio=0.0):
    if max_ratio <= 0: return img_rgb
    h, w = img_rgb.shape[:2]
    dx = int(w * random.uniform(-max_ratio, max_ratio))
    dy = int(h * random.uniform(-max_ratio, max_ratio))
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[0+dx,0+dy],[w-dx,0+dy],[w-dx,h-dy],[0+dx,h-dy]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_rgb, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def pad_to_stride(img, s=32):
    h,w = img.shape[:2]
    H = (h + s - 1) // s * s
    W = (w + s - 1) // s * s
    if (H,W) == (h,w): return img
    out = np.full((H,W,3), 255, np.uint8)
    out[:h,:w] = img
    return out

def luminance_penalty(img):
    # expects RGB; coefficients assume OpenCV BGR->RGB already handled
    Y = 0.2126*img[...,2] + 0.7152*img[...,1] + 0.0722*img[...,0]
    mu, sd = float(Y.mean()), float(Y.std())
    pen = 0.0
    if not (90<=mu<=200): pen += abs(mu-145)/145.0
    if not (25<=sd<=95):  pen += abs(sd-60)/60.0
    return pen

def append_txt_line(path, data: dict):
    cols = ["phase","trial","score","preNMS","postNMS","nms_ms",
            "D","gap","rot_jit_deg","scale_jit","pos_jit","contrast_jit",
            "outline_p","outline_px","micro_enable","micro_density"]
    with open(path, "a") as f:
        f.write(" | ".join([f"{k}={data.get(k,'')}" for k in cols]) + "\n")

# ----------------- homography utils -----------------
def warp_bbox(M, w, h):
    corners = np.array([[0,   w-1, w-1, 0],
                        [0,   0,   h-1, h-1],
                        [1,   1,   1,   1]], dtype=np.float64)
    dst = M @ corners
    dst /= dst[2:3, :]
    xs, ys = dst[0], dst[1]
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

def apply_h(M, pts3xN):
    dst = M @ pts3xN
    dst /= dst[2:3, :]
    return dst

def plane_to_camera_with_anchor(img_plane, H_plane_to_cam, anchor_edge="bottom",
                                max_size=(4096,4096), pad=1.0):
    """
    Warps a fronto-parallel plane image into a tilted camera view using H (plane->camera).
    Scales so the chosen 'near' edge keeps SAME pixel length as in the input.
    Returns (img_cam, valid_mask, M=S·T·H).
    """
    h, w = img_plane.shape[:2]
    xmin, xmax, ymin, ymax = warp_bbox(H_plane_to_cam, w, h)
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad

    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0,   1   ]], dtype=np.float64)

    outW_nat = int(np.ceil(xmax - xmin))
    outH_nat = int(np.ceil(ymax - ymin))
    outW_nat = max(outW_nat, 1); outH_nat = max(outH_nat, 1)

    if anchor_edge.lower() == "bottom":
        plane_edge = np.array([[0, w-1],[h-1, h-1],[1,1]], dtype=np.float64)
        target_len = float(w)
    elif anchor_edge.lower() == "left":
        plane_edge = np.array([[0,0],[0,h-1],[1,1]], dtype=np.float64)
        target_len = float(h)
    else:
        raise ValueError("anchor_edge must be 'bottom' or 'left'.")

    cam_edge_no_scale = apply_h(T @ H_plane_to_cam, plane_edge)
    d = np.linalg.norm(cam_edge_no_scale[:2, 1] - cam_edge_no_scale[:2, 0]) or 1e-9
    s_edge = target_len / d

    MAX_W, MAX_H = max_size
    s_cap = min(MAX_W / outW_nat, MAX_H / outH_nat, 1.0)
    s = min(s_edge, s_cap)

    S = np.array([[s,0,0],[0,s,0],[0,0,1]], dtype=np.float64)
    outW = int(round(outW_nat * s)); outH = int(round(outH_nat * s))
    outW = max(outW,1); outH = max(outH,1)

    M = S @ T @ H_plane_to_cam
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    img_cam = cv2.warpPerspective(img_plane, M, (outW, outH),
                                  flags=interp, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255,255,255))

    mask_plane = np.full((h, w), 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(mask_plane, M, (outW, outH),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_cam, valid_mask, M

def jitter_H(H, ang_deg_sigma=1.2, tx_sigma=3.0, ty_sigma=3.0, s_sigma=0.006):
    ax = np.deg2rad(np.random.randn()*ang_deg_sigma)
    ay = np.deg2rad(np.random.randn()*ang_deg_sigma)
    sx = 1.0 + np.random.randn()*s_sigma
    sy = 1.0 + np.random.randn()*s_sigma
    tx = np.random.randn()*tx_sigma
    ty = np.random.randn()*ty_sigma
    J = np.array([[sx*np.cos(ax), np.sin(ay), tx],
                  [-np.sin(ay),   sy*np.cos(ax), ty],
                  [0, 0, 1]], dtype=np.float64)
    return J @ H

# ----------------- projector pre-compensation -----------------
def precompensate_projector_gamma(img_rgb, proj_gamma=2.2):
    x = (img_rgb.astype(np.float32)/255.0) ** (1.0/proj_gamma)
    return np.clip(x*255,0,255).astype(np.uint8)

# ----------------- RGBA-safe color / contrast -----------------
def random_colorize_rgba(im_rgba, sat=(0.6, 1.2), val=(0.7, 1.1), p=0.7):
    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    if np.random.rand() >= p:
        return im_rgba
    r, g, b, a = im_rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    arr = np.array(rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= np.random.uniform(*sat)
    hsv[...,2] *= np.random.uniform(*val)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out = Image.fromarray(out_rgb).convert("RGBA")
    out.putalpha(a)
    return out

def tint_rgba_numpy(im_rgba, amt):
    if im_rgba.mode != "RGBA":
        im_rgba = im_rgba.convert("RGBA")
    arr = np.array(im_rgba).astype(np.float32)
    rgb = arr[..., :3]; alpha = arr[..., 3:4]
    a = float(abs(amt))
    if amt > 0:
        rgb = rgb*(1.0 - a) + 255.0*a
    else:
        rgb = rgb*(1.0 - a)
    rgb = np.clip(rgb, 0, 255)
    out = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")

# ----------------- pattern rendering (generic) -----------------
def add_micro_dots_rgba(canvas_rgba, density=0.0):
    if density <= 0.0: return canvas_rgba
    arr = np.array(canvas_rgba)
    h, w = arr.shape[:2]
    n = int(w*h*density)
    if n <= 0: return canvas_rgba
    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)
    arr[ys, xs, :3] = 0  # black dots
    arr[ys, xs, 3] = 255
    return Image.fromarray(arr, mode="RGBA")

def render_pattern(tiles, cfg, W=1920, H=1080, bg=(255,255,255,255)):
    D = int(cfg["D"]); GAP = int(cfg["gap"])
    ROT = float(cfg["rot_jit_deg"]); SCALE = float(cfg["scale_jit"])
    PJ = int(cfg["pos_jit"]); CJ = float(cfg["contrast_jit"])
    OUTP = float(cfg.get("outline_p", 0.5))
    OUTPX = int(cfg.get("outline_px", 1))
    MICRO_EN = int(cfg.get("micro_enable", 0))
    MICRO_D = float(cfg.get("micro_density", 0.0))

    step = D + GAP
    cols = max(1, W // step); rows = max(1, H // step)
    canvas = Image.new("RGBA", (W, H), bg)

    for r in range(rows):
        for c in range(cols):
            base = random.choice(tiles).convert("RGBA")
            s = max(8, int(round(D * (1.0 + random.uniform(-SCALE, SCALE)))))
            im = base.resize((s, s), Image.LANCZOS)
            im = random_colorize_rgba(im)
            if CJ > 0:
                amt = random.uniform(-CJ, CJ)
                im = tint_rgba_numpy(im, amt)
            if OUTP > 0 and random.random() < OUTP:
                draw = ImageDraw.Draw(im)
                w_, h_ = im.size
                for k in range(max(1, OUTPX)):
                    draw.rectangle([k, k, w_-1-k, h_-1-k], outline=(0,0,0,255))
            if ROT > 0:
                im = im.rotate(random.uniform(-ROT, ROT), resample=Image.BICUBIC, expand=True)
                im = im.convert("RGBA")
            x = c*step + random.randint(-PJ, PJ) + (D - im.width)//2
            y = r*step + random.randint(-PJ, PJ) + (D - im.height)//2
            canvas.alpha_composite(im, (x, y))

    if MICRO_EN == 1:
        canvas = add_micro_dots_rgba(canvas, density=MICRO_D)

    canvas_rgb = np.array(canvas.convert("RGB"))
    return Image.fromarray(canvas_rgb)

# ----------------- YOLO evaluation -----------------
def load_model(weights, device=""):
    dev = select_device(device)
    m = DetectMultiBackend(weights, device=dev)
    return m

def yolo_eval(model, img_rgb, pre_tau=0.10, post_tau=0.25, iou_thres=0.50, max_det=30000):
    import torch
    s = int(getattr(model, "stride", 32)) or 32
    h0, w0 = img_rgb.shape[:2]
    new_h = int(math.ceil(h0 / s) * s)
    new_w = int(math.ceil(w0 / s) * s)
    im = letterbox(img_rgb, new_shape=(new_h, new_w), stride=s, auto=False)[0]
    im = im.transpose((2, 0, 1))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(model.device).float() / 255.0
    if im.ndim == 3: im = im.unsqueeze(0)

    with torch.no_grad():
        pred = model(im, augment=False, visualize=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    p = pred[0] if pred.ndim == 3 else pred

    obj = p[:, 4:5]; cls = p[:, 5:]
    cls_max = cls.max(dim=1, keepdim=True)[0]
    conf = (obj * cls_max).squeeze(1)
    pre_count = int((conf > pre_tau).sum().item())

    t0 = time.perf_counter()
    det = non_max_suppression(pred, conf_thres=post_tau, iou_thres=iou_thres, max_det=max_det)[0]
    nms_ms = (time.perf_counter() - t0) * 1000.0
    post_count = 0 if det is None else det.shape[0]

    return dict(preNMS=pre_count, postNMS=post_count, nms_ms=nms_ms)

def score_from_metrics(m, w):
    return w["preNMS"]*m["preNMS"] + w["postNMS"]*m["postNMS"] + w["nms_ms"]*m["nms_ms"]

# ----------------- Geometry+optics simulator (PLANE -> CAMERA) -----------------
def homography_sim_then_optics(img_plane_rgb, H, anchor_edge="bottom",
                               max_w=4096, max_h=4096,
                               gamma_range=(0.9, 1.15), blur_sigma=(0.8,1.4),
                               noise_sigma=(1.5,3.5), wb_jit=(0.97,1.03),
                               tiny_persp=0.0, robust_jitter=False):
    H_use = jitter_H(H) if robust_jitter else H
    cam_like, valid_mask, M = plane_to_camera_with_anchor(
        img_plane_rgb, H_use, anchor_edge=anchor_edge, max_size=(max_w, max_h), pad=1.0
    )
    out = apply_gamma(cam_like, random.uniform(*gamma_range))
    out = gaussian_blur(out, random.uniform(*blur_sigma))
    out = add_noise(out, random.uniform(*noise_sigma))
    wr = random.uniform(*wb_jit); wg = random.uniform(*wb_jit); wb = random.uniform(*wb_jit)
    out = apply_wb(out, wr, wg, wb)
    out = small_perspective(out, tiny_persp)
    return out, valid_mask, M

def eval_robust(model, img_plane_rgb, H, args, K=3):
    scores=[]; pre=[]; post=[]; nmsm=[]
    for _ in range(K):
        ae = args.anchor_edge
        if args.robust_jitter and np.random.rand() < 0.10:
            ae = "left" if ae == "bottom" else "bottom"
        sim, valid_mask, _M = homography_sim_then_optics(
            img_plane_rgb, H, anchor_edge=ae,
            max_w=args.max_out_w, max_h=args.max_out_h,
            gamma_range=(0.9, 1.15), blur_sigma=(0.8,1.4),
            noise_sigma=(1.5,3.5), wb_jit=(0.97,1.03),
            tiny_persp=0.0, robust_jitter=args.robust_jitter
        )
        sim = pad_to_stride(sim, 32)
        met = yolo_eval(model, sim, args.pre_tau, args.post_tau, args.iou_thres, args.max_det)
        vis = (valid_mask>0).mean()
        sc = score_from_metrics(met, DEFAULTS["score_weights"]) - 500.0*luminance_penalty(img_plane_rgb) - 300.0*max(0.0, 0.90 - vis)
        scores.append(sc); pre.append(met["preNMS"]); post.append(met["postNMS"]); nmsm.append(met["nms_ms"])
    return dict(score=float(np.mean(scores)),
                preNMS=int(np.mean(pre)), postNMS=int(np.mean(post)), nms_ms=float(np.mean(nmsm)))

# ----------------- CAMERA-SPACE path helpers -----------------
def lock_M_for_plane(H, plane_w, plane_h, anchor_edge, max_w, max_h, pad=0.0):
    """Lock the full warp M=STH and return camera dims. pad lowered to 0.0 to boost coverage."""
    dummy = np.full((plane_h, plane_w, 3), 255, np.uint8)
    cam_like, valid_mask, M = plane_to_camera_with_anchor(
        dummy, H, anchor_edge=anchor_edge, max_size=(max_w, max_h), pad=pad
    )
    outH, outW = cam_like.shape[:2]
    return M, outW, outH

def warp_mask_plane_to_cam(plane_h, plane_w, M, outW, outH):
    mask_plane = np.full((plane_h, plane_w), 255, np.uint8)
    valid_mask = cv2.warpPerspective(mask_plane, M, (outW, outH),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return valid_mask

def simulate_optics_only(img_cam_rgb,
                         gamma_range=(0.9, 1.15), blur_sigma=(0.8,1.4),
                         noise_sigma=(1.5,3.5), wb_jit=(0.97,1.03)):
    out = apply_gamma(img_cam_rgb, random.uniform(*gamma_range))
    out = gaussian_blur(out, random.uniform(*blur_sigma))
    out = add_noise(out, random.uniform(*noise_sigma))
    wr = random.uniform(*wb_jit); wg = random.uniform(*wb_jit); wb = random.uniform(*wb_jit)
    out = apply_wb(out, wr, wg, wb)
    return out

def eval_camera_design(model, cam_canvas_rgb, M, M_inv, plane_w, plane_h, outW, outH, args, K=3):
    """
    1) Back-map camera design to plane via M^-1 (The projected pattern on the ground).
    2) Forward with the SAME M to camera; apply photometric jitter.
    3) Score with YOLO in camera space; penalties on plane luminance and camera coverage.
    """
    scores=[]; pre=[]; post=[]; nmsm=[]
    # Coverage is constant for fixed geometry; compute once
    valid_mask = warp_mask_plane_to_cam(plane_h, plane_w, M, outW, outH)
    vis = (valid_mask>0).mean()
    for _ in range(K):
        plane_img = cv2.warpPerspective(cam_canvas_rgb, M_inv, (plane_w, plane_h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        lum_pen = luminance_penalty(plane_img)
        cam_warped = cv2.warpPerspective(plane_img, M, (outW, outH),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        cam_sim = simulate_optics_only(cam_warped)

        cam_sim = pad_to_stride(cam_sim, 32)
        met = yolo_eval(model, cam_sim, args.pre_tau, args.post_tau, args.iou_thres, args.max_det)

        # softer coverage target (0.85) for camera mode
        sc = score_from_metrics(met, DEFAULTS["score_weights"]) - 500.0*lum_pen - 300.0*max(0.0, 0.85 - vis)
        scores.append(sc); pre.append(met["preNMS"]); post.append(met["postNMS"]); nmsm.append(met["nms_ms"])
    return dict(score=float(np.mean(scores)),
                preNMS=int(np.mean(pre)), postNMS=int(np.mean(post)), nms_ms=float(np.mean(nmsm)))

# ----------------- Phase A/B (PLANE-SPACE) -----------------
def phaseA_plane(tiles, model, args, H):
    rows = []; best = []
    total_cells = max(1, len(args.D_list) * len(args.gap_list))
    per_combo = max(1, args.phaseA_trials // total_cells)

    plane_dir = Path(args.out_dir)/"plane"; plane_dir.mkdir(parents=True, exist_ok=True)
    warped_dir = Path(args.out_dir)/"warped"; warped_dir.mkdir(parents=True, exist_ok=True)
    stream_path = Path(args.out_dir)/"phaseA_plane_stream.txt"

    for D in args.D_list:
        for g in args.gap_list:
            for _ in range(per_combo):
                cfg = dict(
                    D=int(D),
                    gap=int(g),
                    rot_jit_deg=random.uniform(0.0, args.rot_jit_max),
                    scale_jit=random.uniform(args.scale_jit_min, args.scale_jit_max),
                    pos_jit=random.randint(0, args.pos_jit_max),
                    contrast_jit=random.uniform(0.0, args.contrast_jit_max),
                    outline_p=random.uniform(0.2, 0.9),
                    outline_px=random.randint(1, 2),
                    micro_enable=random.choice([0,1]),
                    micro_density=random.uniform(0.0008, 0.003),
                )
                plane_img = render_pattern(tiles, cfg, args.plane_w, args.plane_h)
                plane_np = np.array(plane_img)
                plane_np = cv2.GaussianBlur(plane_np, (0,0), sigmaX=0.6, sigmaY=0.6, borderType=cv2.REPLICATE)

                m = eval_robust(model, plane_np, H, args, K=args.K)
                s = m["score"]
                row = dict(phase="A_plane", **m, **cfg)
                rows.append(row); append_txt_line(stream_path, row)

                if len(best) < args.save_top_k or s > best[0][0]:
                    sim_one, _, _ = homography_sim_then_optics(
                        plane_np, H, anchor_edge=args.anchor_edge,
                        max_w=args.max_out_w, max_h=args.max_out_h,
                        tiny_persp=0.0, robust_jitter=False
                    )
                    stamp = f"D{cfg['D']}_g{cfg['gap']}_r{cfg['rot_jit_deg']:.2f}_s{cfg['scale_jit']:.2f}_p{cfg['pos_jit']}_c{cfg['contrast_jit']:.3f}_score{s:.1f}"
                    Image.fromarray(plane_np).save(plane_dir/f"A_{stamp}.jpg", quality=95)
                    Image.fromarray(sim_one).save(warped_dir/f"A_{stamp}.jpg", quality=95)
                    best.append((s, stamp)); best = sorted(best, key=lambda x: x[0])[-args.save_top_k:]
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir)/"phaseA_plane_results.txt", sep="\t", index=False)
    return df

def phaseB_optuna_plane(tiles, model, args, warm_df, H):
    try:
        import optuna
    except Exception:
        print("Optuna not installed; skipping Phase B (plane). Install via:  pip install optuna")
        return pd.DataFrame()

    def suggest_cfg(trial):
        micro_enable = trial.suggest_categorical("micro_enable", [0,1])
        micro_density = trial.suggest_float("micro_density", 0.0008, 0.003) if micro_enable else 0.0
        return dict(
            D=trial.suggest_categorical("D", args.D_list),
            gap=trial.suggest_categorical("gap", args.gap_list),
            rot_jit_deg=trial.suggest_float("rot_jit_deg", 0.0, args.rot_jit_max),
            scale_jit=trial.suggest_float("scale_jit", args.scale_jit_min, args.scale_jit_max),
            pos_jit=trial.suggest_int("pos_jit", 0, args.pos_jit_max),
            contrast_jit=trial.suggest_float("contrast_jit", 0.0, args.contrast_jit_max),
            outline_p=trial.suggest_float("outline_p", 0.2, 0.9),
            outline_px=trial.suggest_int("outline_px", 1, 2),
            micro_enable=micro_enable,
            micro_density=micro_density,
        )

    rows = []; best = []
    stream_path = Path(args.out_dir)/"phaseB_plane_stream.txt"

    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, n_startup_trials=20)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    warm_top = warm_df.sort_values("score", ascending=False).head(min(20, len(warm_df)))
    for _, r in warm_top.iterrows():
        ws = dict(
            D=int(r.D), gap=int(r.gap),
            rot_jit_deg=float(r.rot_jit_deg),
            scale_jit=float(r.scale_jit),
            pos_jit=int(r.pos_jit),
            contrast_jit=float(r.contrast_jit),
            outline_p=float(r.get("outline_p", 0.5)),
            outline_px=int(r.get("outline_px", 1)),
            micro_enable=int(r.get("micro_enable", 0)),
            micro_density=float(r.get("micro_density", 0.0015))
        )
        study.enqueue_trial(ws)

    def objective(trial):
        cfg = suggest_cfg(trial)
        plane_img = render_pattern(tiles, cfg, args.plane_w, args.plane_h)
        plane_np = np.array(plane_img)
        plane_np = cv2.GaussianBlur(plane_np, (0,0), sigmaX=0.6, sigmaY=0.6, borderType=cv2.REPLICATE)

        K_use = 1 if trial.number < 30 else args.K
        m = eval_robust(model, plane_np, H, args, K=K_use)
        s = m["score"]
        row = dict(phase="B_plane", trial=trial.number, **m, **cfg)
        rows.append(row); append_txt_line(stream_path, row)

        if luminance_penalty(plane_np) > 0.35:
            trial.report(-1e9, step=0); raise optuna.TrialPruned()

        trial.report(s, step=0)
        if trial.should_prune(): raise optuna.TrialPruned()

        nonlocal best
        if len(best) < args.save_top_k or s > best[0][0]:
            sim_one, _, _ = homography_sim_then_optics(
                plane_np, H, anchor_edge=args.anchor_edge,
                max_w=args.max_out_w, max_h=args.max_out_h,
                tiny_persp=0.0, robust_jitter=False
            )
            stamp = f"D{cfg['D']}_g{cfg['gap']}_r{cfg['rot_jit_deg']:.2f}_s{cfg['scale_jit']:.2f}_p{cfg['pos_jit']}_c{cfg['contrast_jit']:.3f}_score{s:.1f}"
            Image.fromarray(plane_np).save(Path(args.out_dir)/"plane"/f"B_{stamp}.jpg", quality=95)
            Image.fromarray(sim_one).save(Path(args.out_dir)/"warped"/f"B_{stamp}.jpg", quality=95)
            best.append((s, stamp)); best = sorted(best, key=lambda x: x[0])[-args.save_top_k:]
        return s

    study.optimize(objective, n_trials=args.phaseB_trials, n_jobs=1, show_progress_bar=False)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir)/"phaseB_plane_results.txt", sep="\t", index=False)
    return df

# ----------------- Phase A (CAMERA-SPACE) -----------------
def phaseA_camera(tiles, model, args, M, M_inv, outW, outH):
    rows=[]; best=[]
    total_cells = max(1, len(args.D_list) * len(args.gap_list))
    per_combo = max(1, args.phaseA_trials // total_cells)

    cam_dir = Path(args.out_dir)/"camera_designs"; cam_dir.mkdir(parents=True, exist_ok=True)
    plane_dir = Path(args.out_dir)/"plane_from_camera"; plane_dir.mkdir(parents=True, exist_ok=True)
    warped_dir = Path(args.out_dir)/"warped_from_camera"; warped_dir.mkdir(parents=True, exist_ok=True)
    stream_path = Path(args.out_dir)/"phaseA_camera_stream.txt"

    for D in args.D_list:
        for g in args.gap_list:
            for _ in range(per_combo):
                cfg = dict(
                    D=int(D),
                    gap=int(g),
                    rot_jit_deg=random.uniform(0.0, args.rot_jit_max),
                    scale_jit=random.uniform(args.scale_jit_min, args.scale_jit_max),
                    pos_jit=random.randint(0, args.pos_jit_max),
                    contrast_jit=random.uniform(0.0, args.contrast_jit_max),
                    outline_p=random.uniform(0.2, 0.9),
                    outline_px=random.randint(1, 2),
                    micro_enable=random.choice([0,1]),
                    micro_density=random.uniform(0.0008, 0.003),
                )
                cam_canvas = render_pattern(tiles, cfg, outW, outH)
                cam_np = np.array(cam_canvas)

                m = eval_camera_design(model, cam_np, M, M_inv,
                                       args.plane_w, args.plane_h, outW, outH, args, K=args.K)
                s = m["score"]
                row = dict(phase="A_camera", **m, **cfg)
                rows.append(row); append_txt_line(stream_path, row)

                if len(best) < args.save_top_k or s > best[0][0]:
                    plane_img = cv2.warpPerspective(cam_np, M_inv, (args.plane_w, args.plane_h),
                                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
                    plane_for_export = precompensate_projector_gamma(plane_img, proj_gamma=2.2)
                    cam_sim = cv2.warpPerspective(plane_img, M, (outW, outH),
                                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

                    stamp = f"CAM_D{cfg['D']}_g{cfg['gap']}_r{cfg['rot_jit_deg']:.2f}_s{cfg['scale_jit']:.2f}_p{cfg['pos_jit']}_c{cfg['contrast_jit']:.3f}_score{s:.1f}"
                    Image.fromarray(cam_np).save(cam_dir/f"A_{stamp}.jpg", quality=95)
                    Image.fromarray(plane_for_export).save(plane_dir/f"A_{stamp}.jpg", quality=95)
                    Image.fromarray(cam_sim).save(warped_dir/f"A_{stamp}.jpg", quality=95)
                    best.append((s, stamp)); best = sorted(best, key=lambda x: x[0])[-args.save_top_k:]
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir)/"phaseA_camera_results.txt", sep="\t", index=False)
    return df

# ----------------- Phase B (CAMERA-SPACE, relaxed gates) -----------------
def phaseB_optuna_camera(tiles, model, args, warm_df, M, M_inv, outW, outH):
    """
    Bayesian optimization in CAMERA space (design in camera pixels).
    Warm-start from top Phase-A camera results.
    - Coverage is a SOFT penalty only.
    - Luminance prune relaxed (0.55).
    - No pruner to ensure early learning (switch later if desired).
    """
    try:
        import optuna
    except Exception:
        print("Optuna not installed; skipping Phase B (camera). Install via:  pip install optuna")
        return pd.DataFrame()

    def suggest_cfg(trial):
        micro_enable = trial.suggest_categorical("micro_enable", [0,1])
        micro_density = trial.suggest_float("micro_density", 0.0008, 0.003) if micro_enable else 0.0
        return dict(
            D=trial.suggest_categorical("D", args.D_list),
            gap=trial.suggest_categorical("gap", args.gap_list),
            rot_jit_deg=trial.suggest_float("rot_jit_deg", 0.0, args.rot_jit_max),
            scale_jit=trial.suggest_float("scale_jit", args.scale_jit_min, args.scale_jit_max),
            pos_jit=trial.suggest_int("pos_jit", 0, args.pos_jit_max),
            contrast_jit=trial.suggest_float("contrast_jit", 0.0, args.contrast_jit_max),
            outline_p=trial.suggest_float("outline_p", 0.2, 0.9),
            outline_px=trial.suggest_int("outline_px", 1, 2),
            micro_enable=micro_enable,
            micro_density=micro_density,
        )

    rows=[]; best=[]
    stream_path = Path(args.out_dir)/"phaseB_camera_stream.txt"

    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, n_startup_trials=20)
    pruner = optuna.pruners.NopPruner()  # <- relaxed; guarantees some completed trials
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    warm_top = warm_df.sort_values("score", ascending=False).head(min(20, len(warm_df)))
    for _, r in warm_top.iterrows():
        ws = dict(
            D=int(r.D), gap=int(r.gap),
            rot_jit_deg=float(r.rot_jit_deg),
            scale_jit=float(r.scale_jit),
            pos_jit=int(r.pos_jit),
            contrast_jit=float(r.contrast_jit),
            outline_p=float(r.get("outline_p", 0.5)),
            outline_px=int(r.get("outline_px", 1)),
            micro_enable=int(r.get("micro_enable", 0)),
            micro_density=float(r.get("micro_density", 0.0015)),
        )
        study.enqueue_trial(ws)

    # constant coverage for given geometry (log once)
    vis_const = (warp_mask_plane_to_cam(args.plane_h, args.plane_w, M, outW, outH) > 0).mean()
    print(f"[camera PhaseB] locked geometry coverage vis={vis_const:.3f}")

    def objective(trial):
        cfg = suggest_cfg(trial)
        cam_canvas = render_pattern(tiles, cfg, outW, outH)
        cam_np = np.array(cam_canvas)

        # soft checks only: luminance prune relaxed; no coverage prune
        plane_img = cv2.warpPerspective(cam_np, M_inv, (args.plane_w, args.plane_h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        lum_pen = luminance_penalty(plane_img)
        if lum_pen > 0.55:
            trial.report(-1e9, step=0); raise optuna.TrialPruned()

        # multi-fidelity: cheap K=1 early, full K later
        K_use = 1 if trial.number < 30 else args.K
        m = eval_camera_design(model, cam_np, M, M_inv, args.plane_w, args.plane_h, outW, outH, args, K=K_use)
        s = m["score"]

        row = dict(phase="B_camera", trial=trial.number, **m, **cfg)
        rows.append(row); append_txt_line(stream_path, row)

        trial.report(s, step=0)

        nonlocal best
        if len(best) < args.save_top_k or s > best[0][0]:
            cam_sim = cv2.warpPerspective(plane_img, M, (outW, outH),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            plane_for_export = precompensate_projector_gamma(plane_img, proj_gamma=2.2)
            stamp = f"CAM_D{cfg['D']}_g{cfg['gap']}_r{cfg['rot_jit_deg']:.2f}_s{cfg['scale_jit']:.2f}_p{cfg['pos_jit']}_c{cfg['contrast_jit']:.3f}_score{s:.1f}"
            Path(args.out_dir,"camera_designs").mkdir(parents=True, exist_ok=True)
            Path(args.out_dir,"plane_from_camera").mkdir(parents=True, exist_ok=True)
            Path(args.out_dir,"warped_from_camera").mkdir(parents=True, exist_ok=True)
            Image.fromarray(cam_np).save(Path(args.out_dir)/"camera_designs"/f"B_{stamp}.jpg", quality=95)
            Image.fromarray(plane_for_export).save(Path(args.out_dir)/"plane_from_camera"/f"B_{stamp}.jpg", quality=95)
            Image.fromarray(cam_sim).save(Path(args.out_dir)/"warped_from_camera"/f"B_{stamp}.jpg", quality=95)
            best.append((s, stamp)); best = sorted(best, key=lambda x: x[0])[-args.save_top_k:]
        return s

    study.optimize(objective, n_trials=args.phaseB_trials, n_jobs=1, show_progress_bar=False)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir)/"phaseB_camera_results.txt", sep="\t", index=False)
    return df

# ----------------- Export top-k PLANE images -----------------
def export_topk_planes_from_df(df, tiles, args, topk, H=None, M=None, M_inv=None, outW=None, outH=None, mode="plane"):
    dst_plane = Path(args.out_dir)/"to_project"; dst_plane.mkdir(parents=True, exist_ok=True)
    dst_warped = Path(args.out_dir)/"to_project_warped"; dst_warped.mkdir(parents=True, exist_ok=True)
    best = df.sort_values("score", ascending=False).head(min(topk, len(df))).reset_index(drop=True)

    lines = []
    for i, r in best.iterrows():
        cfg = dict(
            D=int(r.D),
            gap=int(r.gap),
            rot_jit_deg=float(r.rot_jit_deg),
            scale_jit=float(r.scale_jit),
            pos_jit=int(r.pos_jit),
            contrast_jit=float(r.contrast_jit),
            outline_p=float(r.get("outline_p", 0.5)),
            outline_px=int(r.get("outline_px", 1)),
            micro_enable=int(r.get("micro_enable", 0)),
            micro_density=float(r.get("micro_density", 0.0015)),
        )

        if mode == "plane":
            plane = render_pattern(tiles, cfg, args.plane_w, args.plane_h)
            plane_np = np.array(plane)
            warped, _, _ = homography_sim_then_optics(
                plane_np, H, anchor_edge=args.anchor_edge,
                max_w=args.max_out_w, max_h=args.max_out_h,
                tiny_persp=0.0, robust_jitter=False
            )
            tag = f"rank{i+1}_score{r.score:.1f}_D{cfg['D']}_g{cfg['gap']}"
            Image.fromarray(precompensate_projector_gamma(plane_np, proj_gamma=2.2)).save(dst_plane/f"{tag}.jpg", quality=95)
            Image.fromarray(warped).save(dst_warped/f"{tag}.jpg", quality=95)
            lines.append(f"{tag} | cfg={cfg}")
        else:
            cam_canvas = render_pattern(tiles, cfg, outW, outH)
            cam_np = np.array(cam_canvas)
            plane_np = cv2.warpPerspective(cam_np, M_inv, (args.plane_w, args.plane_h),
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            warped = cv2.warpPerspective(plane_np, M, (outW, outH),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            tag = f"CAM_rank{i+1}_score{r.score:.1f}_D{cfg['D']}_g{cfg['gap']}"
            Image.fromarray(precompensate_projector_gamma(plane_np, proj_gamma=2.2)).save(dst_plane/f"{tag}.jpg", quality=95)
            Image.fromarray(warped).save(dst_warped/f"{tag}.jpg", quality=95)
            lines.append(f"{tag} | cfg={cfg}")

    with open(Path(args.out_dir)/"to_project_params.txt", "w") as f:
        f.write("\n".join(lines))

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["plane","camera"], default="camera",
                    help="Optimization domain: plane (original) or camera (new).")
    ap.add_argument("--tiles_dir", type=str, required=True)
    ap.add_argument("--weights", type=str, default="yolov5s.pt")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="runs/pattern_opt_dualspace")

    ap.add_argument("--phaseA_trials", type=int, default=500)
    ap.add_argument("--phaseB_trials", type=int, default=2500)
    ap.add_argument("--save_top_k", type=int, default=DEFAULTS["save_top_k"])

    # search space
    ap.add_argument("--D_list", type=int, nargs="+", default=[96,112,128,144,160])
    ap.add_argument("--gap_list", type=int, nargs="+", default=[0,1,2,3])
    ap.add_argument("--rot_jit_max", type=float, default=5.0)
    ap.add_argument("--scale_jit_min", type=float, default=0.06)
    ap.add_argument("--scale_jit_max", type=float, default=0.15)
    ap.add_argument("--pos_jit_max", type=int, default=10)
    ap.add_argument("--contrast_jit_max", type=float, default=0.10)

    # thresholds
    ap.add_argument("--pre_tau", type=float, default=DEFAULTS["pre_tau"])
    ap.add_argument("--post_tau", type=float, default=DEFAULTS["post_tau"])
    ap.add_argument("--iou_thres", type=float, default=DEFAULTS["iou_thres"])
    ap.add_argument("--max_det", type=int, default=DEFAULTS["max_det"])

    # geometry / sizes
    ap.add_argument("--anchor_edge", type=str, default="bottom", choices=["bottom","left"])
    ap.add_argument("--max_out_w", type=int, default=4096)
    ap.add_argument("--max_out_h", type=int, default=4096)
    ap.add_argument("--plane_w", type=int, default=1920)
    ap.add_argument("--plane_h", type=int, default=1080)

    ap.add_argument("--robust_jitter", action="store_true", help="(plane mode) tiny H jitters for robustness")
    ap.add_argument("--K", type=int, default=3, help="robust evaluation repeats per trial")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(0)
    tiles, _ = load_tiles(args.tiles_dir)
    model = load_model(args.weights, args.device)

    # Save config + H
    cfg = vars(args).copy()
    cfg["H_hardcoded"] = H_CONST.tolist()
    with open(Path(args.out_dir)/"config.json", "w") as f: json.dump(cfg, f, indent=2)

    if args.mode == "plane":
        H = H_CONST.copy()
        dfA = phaseA_plane(tiles, model, args, H)
        print("Phase A (plane) complete. Top-5:\n", dfA.sort_values("score", ascending=False).head(5))
        dfB = phaseB_optuna_plane(tiles, model, args, dfA, H)
        dfs = [df for df in (dfA, dfB) if len(df) > 0]
        df = pd.concat(dfs, ignore_index=True) if dfs else dfA
        df.sort_values("score", ascending=False).to_csv(Path(args.out_dir)/"all_results_ranked.txt", sep="\t", index=False)
        export_topk_planes_from_df(df, tiles, args, topk=10, H=H, mode="plane")
        print("Exported top-10 plane patterns to:", Path(args.out_dir)/"to_project")
        return

    # -------------- CAMERA mode --------------
    M, outW, outH = lock_M_for_plane(H_CONST, args.plane_w, args.plane_h,
                                     args.anchor_edge, args.max_out_w, args.max_out_h, pad=0.0)
    M_inv = np.linalg.inv(M)
    np.save(Path(args.out_dir)/"locked_M.npy", M)
    with open(Path(args.out_dir)/"locked_M_meta.json", "w") as f:
        json.dump(dict(outW=outW, outH=outH, plane_w=args.plane_w, plane_h=args.plane_h), f, indent=2)

    vis_probe = (warp_mask_plane_to_cam(args.plane_h, args.plane_w, M, outW, outH) > 0).mean()
    print(f"[camera-mode] locked M size=({outW}x{outH}) from plane ({args.plane_w}x{args.plane_h}); coverage vis={vis_probe:.3f}")

    dfA_cam = phaseA_camera(tiles, model, args, M, M_inv, outW, outH)
    print("Phase A (camera) complete. Top-5:\n", dfA_cam.sort_values("score", ascending=False).head(5))

    dfB_cam = phaseB_optuna_camera(tiles, model, args, dfA_cam, M, M_inv, outW, outH)
    dfs = [df for df in (dfA_cam, dfB_cam) if len(df) > 0]
    df = pd.concat(dfs, ignore_index=True) if dfs else dfA_cam
    df.sort_values("score", ascending=False).to_csv(Path(args.out_dir)/"all_results_ranked.txt", sep="\t", index=False)

    export_topk_planes_from_df(df, tiles, args, topk=10, M=M, M_inv=M_inv, outW=outW, outH=outH, mode="camera")
    print("Exported top-10 plane patterns (from camera designs) to:", Path(args.out_dir)/"to_project")

if __name__ == "__main__":
    main()
