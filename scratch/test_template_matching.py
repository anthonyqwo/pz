import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, r"d:\pz\src")

import cv2
from puzzle_recognition.config import DetectorConfig, MatcherConfig
from puzzle_recognition.board_builder import load_board_config
from puzzle_recognition.piece_detector import detect_black_pieces
from puzzle_recognition.shape_matcher import (
    crop_mask, rotate_mask, mask_centroid,
    load_slot_mask, contour_to_mask
)

# Load data
image_path = r"C:\Users\kioki\.gemini\antigravity\brain\28e31ef9-ef68-4527-9a89-ab5861391674\media__1779463900515.jpg"
board_id = "board_photo_skeleton_w3"
board_config = load_board_config(board_id)
image = cv2.imread(image_path)
det_cfg = DetectorConfig(
    gray_threshold=60,
    min_piece_area=800.0,
    max_piece_area=350000.0,
    max_bbox_width=800,
    max_bbox_height=800,
    solidity_max=1.0,
    expected_max_pieces=10,
)
pieces, _ = detect_black_pieces(image, det_cfg)
piece = pieces[0]
slot = board_config["slots"][427]

piece_mask_full = contour_to_mask(piece["contour"], tuple(board_config["rectified_size"]))
slot_mask = load_slot_mask(board_id, slot)

cfg = MatcherConfig()
translations = list(cfg.translation_search)  # [-10, -5, 0, 5, 10]
allow_mirror = bool(cfg.allow_mirror)

piece_crop, _ = crop_mask(piece_mask_full)
slot_crop, _ = crop_mask(slot_mask)

piece_diag = int(np.ceil(np.sqrt(piece_crop.shape[0]**2 + piece_crop.shape[1]**2)))
max_trans = max(abs(t) for t in translations) if translations else 0

local_w = max(slot_crop.shape[1], piece_diag) + 2 * max_trans + 10
local_h = max(slot_crop.shape[0], piece_diag) + 2 * max_trans + 10
local_shape = (local_h, local_w)
local_center = (local_w / 2.0, local_h / 2.0)

# Original local placement
def place_mask_local(mask, canvas_shape, target_cx, target_cy):
    h, w = mask.shape[:2]
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"]:
        cx, cy = float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    else:
        cx, cy = w / 2.0, h / 2.0

    canvas_h, canvas_w = canvas_shape
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    x0 = int(round(target_cx - cx))
    y0 = int(round(target_cy - cy))
    x1 = x0 + w
    y1 = y0 + h

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(canvas_w, x1)
    dst_y1 = min(canvas_h, y1)

    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        result[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return result

local_slot_mask = place_mask_local(slot_crop, local_shape, local_center[0], local_center[1])
rotated_piece = rotate_mask(piece_crop, 45)
local_piece_mask = place_mask_local(rotated_piece, local_shape, local_center[0], local_center[1])

# Apply Gaussian blur if soft IoU is enabled
soft_mask_ksize = 5
local_slot_mask_blurred = cv2.GaussianBlur(local_slot_mask, (soft_mask_ksize, soft_mask_ksize), 0).astype(np.float32) / 255.0
local_piece_mask_blurred = cv2.GaussianBlur(local_piece_mask, (soft_mask_ksize, soft_mask_ksize), 0).astype(np.float32) / 255.0

# 1. Benchmark nested loops
print("Timing Original Loop...")
t0 = time.time()
best_iou_loop = 0.0
best_dx_loop = 0
best_dy_loop = 0

sum_rotated_pixels = local_piece_mask_blurred.sum()
sum_slot_pixels = local_slot_mask_blurred.sum()

H, W = local_shape
for dx in translations:
    for dy in translations:
        if dy >= 0:
            ay0, ay1 = 0, H - dy
            by0, by1 = dy, H
        else:
            ay0, ay1 = -dy, H
            by0, by1 = 0, H + dy

        if dx >= 0:
            ax0, ax1 = 0, W - dx
            bx0, bx1 = dx, W
        else:
            ax0, ax1 = -dx, W
            bx0, bx1 = 0, W + dx

        intersection_sum = np.minimum(
            local_piece_mask_blurred[ay0:ay1, ax0:ax1],
            local_slot_mask_blurred[by0:by1, bx0:bx1]
        ).sum()
        union_sum = sum_rotated_pixels + sum_slot_pixels - intersection_sum
        iou = float(intersection_sum / union_sum) if union_sum > 0 else 0.0
        
        if iou > best_iou_loop:
            best_iou_loop = iou
            best_dx_loop = dx
            best_dy_loop = dy
t1 = time.time()
print(f"Loop: Time={(t1-t0)*1000:.3f}ms, IoU={best_iou_loop:.4f}, dx={best_dx_loop}, dy={best_dy_loop}")


# 2. Benchmark matchTemplate
print("\nTiming cv2.matchTemplate...")
t0 = time.time()

# Pad local_slot_mask_blurred
T_max = max(abs(t) for t in translations)
slot_pad = cv2.copyMakeBorder(local_slot_mask_blurred, T_max, T_max, T_max, T_max, cv2.BORDER_CONSTANT, value=0.0)

# Run matchTemplate
res = cv2.matchTemplate(slot_pad, local_piece_mask_blurred, cv2.TM_CCORR)

# Only evaluate allowed translations
best_iou_tmpl = 0.0
best_dx_tmpl = 0
best_dy_tmpl = 0

for dx in translations:
    for dy in translations:
        r = T_max + dy
        c = T_max + dx
        intersection_sum = res[r, c]
        union_sum = sum_rotated_pixels + sum_slot_pixels - intersection_sum
        iou = float(intersection_sum / union_sum) if union_sum > 0 else 0.0
        
        if iou > best_iou_tmpl:
            best_iou_tmpl = iou
            best_dx_tmpl = dx
            best_dy_tmpl = dy

t1 = time.time()
print(f"matchTemplate: Time={(t1-t0)*1000:.3f}ms, IoU={best_iou_tmpl:.4f}, dx={best_dx_tmpl}, dy={best_dy_tmpl}")
print(f"Speedup: {((t1-t0) / (t1-t0) if t1-t0 == 0 else 1.0) * 100:.2f}% (Wait, let's print absolute times)")
print(f"Loop absolute: {(t1-t0):.6f}s vs matchTemplate absolute: {(t1-t0):.6f}s")
