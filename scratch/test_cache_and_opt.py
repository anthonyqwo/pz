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
    find_candidate_slots, crop_mask, rotate_mask, mask_centroid,
    mirror_variants, contour_to_mask
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
cfg = MatcherConfig()

# In-memory mask cache
mask_cache = {}

def get_cached_slot_mask(board_id, slot):
    cache_key = (board_id, slot["slot_id"])
    if cache_key in mask_cache:
        return mask_cache[cache_key]
    
    path = Path("d:/pz/data/boards") / board_id / slot["mask_path"]
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read slot mask: {path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_cache[cache_key] = binary
    return binary

# Original matching with cached masks
def original_match(piece_mask, slot_mask, cfg):
    coarse_step = int(cfg.angle_step_coarse)
    refine_range = int(cfg.angle_refine_range)
    fine_step = int(cfg.angle_step_fine)
    translations = list(cfg.translation_search)
    allow_mirror = bool(cfg.allow_mirror)

    piece_crop, _ = crop_mask(piece_mask)
    slot_center = mask_centroid(slot_mask)
    best = {
        "iou": 0.0,
        "rotation": 0,
        "dx": 0,
        "dy": 0,
        "mirrored": False,
        "mirror_mode": "none",
        "transform": {},
    }

    try:
        slot_crop, slot_bbox = crop_mask(slot_mask)
    except ValueError:
        return best

    piece_diag = int(np.ceil(np.sqrt(piece_crop.shape[0]**2 + piece_crop.shape[1]**2)))
    max_trans = max(abs(t) for t in translations) if translations else 0

    local_w = max(slot_crop.shape[1], piece_diag) + 2 * max_trans + 10
    local_h = max(slot_crop.shape[0], piece_diag) + 2 * max_trans + 10
    local_shape = (local_h, local_w)
    local_center = (local_w / 2.0, local_h / 2.0)

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
    
    use_soft_iou = bool(cfg.use_soft_iou)
    soft_mask_ksize = int(cfg.soft_mask_ksize)
    if use_soft_iou and soft_mask_ksize > 1:
        if soft_mask_ksize % 2 == 0:
            soft_mask_ksize += 1
        local_slot_mask_for_matching = cv2.GaussianBlur(local_slot_mask, (soft_mask_ksize, soft_mask_ksize), 0)
        sum_slot_pixels = float(local_slot_mask_for_matching.sum())
    else:
        local_slot_mask_for_matching = local_slot_mask
        sum_slot_pixels = int(local_slot_mask.sum() // 255)

    def evaluate(variant_mask, angle, mirror_mode):
        nonlocal best
        rotated = rotate_mask(variant_mask, angle)
        placed_rotated = place_mask_local(rotated, local_shape, local_center[0], local_center[1])
        
        if use_soft_iou and soft_mask_ksize > 1:
            placed_rotated_blurred = cv2.GaussianBlur(placed_rotated, (soft_mask_ksize, soft_mask_ksize), 0)
            sum_rotated_pixels = float(placed_rotated_blurred.sum())
        else:
            placed_rotated_blurred = placed_rotated
            sum_rotated_pixels = int(placed_rotated.sum() // 255)

        if sum_rotated_pixels == 0 or sum_slot_pixels == 0:
            return

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

                if use_soft_iou and soft_mask_ksize > 1:
                    intersection_sum = np.minimum(
                        placed_rotated_blurred[ay0:ay1, ax0:ax1],
                        local_slot_mask_for_matching[by0:by1, bx0:bx1]
                    ).sum()
                    union_sum = sum_rotated_pixels + sum_slot_pixels - intersection_sum
                    iou = float(intersection_sum / union_sum) if union_sum > 0 else 0.0
                else:
                    intersection_sum = (placed_rotated_blurred[ay0:ay1, ax0:ax1] & local_slot_mask_for_matching[by0:by1, bx0:bx1]).sum()
                    intersection_pixels = int(intersection_sum // 255)
                    union_pixels = sum_rotated_pixels + sum_slot_pixels - intersection_pixels
                    iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0

                if iou > best["iou"]:
                    best = {
                        "iou": iou,
                        "rotation": int(angle % 360),
                        "dx": int(dx),
                        "dy": int(dy),
                        "mirrored": mirror_mode != "none",
                        "mirror_mode": mirror_mode,
                        "transform": {
                            "slot_center": [float(slot_center[0]), float(slot_center[1])],
                            "rotated_shape": [int(rotated.shape[1]), int(rotated.shape[0])],
                        },
                    }

    for mirror_mode, variant_mask in mirror_variants(piece_crop, allow_mirror):
        for angle in range(0, 360, coarse_step):
            evaluate(variant_mask, angle, mirror_mode)

    refine_angles = range(int(best["rotation"]) - refine_range, int(best["rotation"]) + refine_range + 1, max(1, fine_step))
    for mirror_mode, variant_mask in mirror_variants(piece_crop, allow_mirror):
        for angle in refine_angles:
            evaluate(variant_mask, angle % 360, mirror_mode)

    return best

# Optimized matching with pre-aligned centering and two-stage Coarse-to-Fine Search
def optimized_match(piece_mask, slot_mask, cfg):
    coarse_step = int(cfg.angle_step_coarse)
    refine_range = int(cfg.angle_refine_range)
    fine_step = int(cfg.angle_step_fine)
    translations = list(cfg.translation_search)
    allow_mirror = bool(cfg.allow_mirror)

    piece_crop, _ = crop_mask(piece_mask)
    slot_center = mask_centroid(slot_mask)
    best = {
        "iou": 0.0,
        "rotation": 0,
        "dx": 0,
        "dy": 0,
        "mirrored": False,
        "mirror_mode": "none",
        "transform": {},
    }

    try:
        slot_crop, slot_bbox = crop_mask(slot_mask)
    except ValueError:
        return best

    piece_diag = int(np.ceil(np.sqrt(piece_crop.shape[0]**2 + piece_crop.shape[1]**2)))
    max_trans = max(abs(t) for t in translations) if translations else 0

    local_w = max(slot_crop.shape[1], piece_diag) + 2 * max_trans + 10
    local_h = max(slot_crop.shape[0], piece_diag) + 2 * max_trans + 10
    local_shape = (local_h, local_w)
    local_center = (local_w / 2.0, local_h / 2.0)

    # Pre-align slot mask once
    def place_mask_local_no_moments(mask, canvas_shape, target_cx, target_cy, cx, cy):
        h, w = mask.shape[:2]
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

    slot_moments = cv2.moments((slot_crop > 0).astype(np.uint8))
    slot_cx, slot_cy = (slot_moments["m10"] / slot_moments["m00"], slot_moments["m01"] / slot_moments["m00"]) if slot_moments["m00"] else (slot_crop.shape[1]/2.0, slot_crop.shape[0]/2.0)
    local_slot_mask = place_mask_local_no_moments(slot_crop, local_shape, local_center[0], local_center[1], slot_cx, slot_cy)
    
    use_soft_iou = bool(cfg.use_soft_iou)
    soft_mask_ksize = int(cfg.soft_mask_ksize)
    if use_soft_iou and soft_mask_ksize > 1:
        if soft_mask_ksize % 2 == 0:
            soft_mask_ksize += 1
        local_slot_mask_for_matching = cv2.GaussianBlur(local_slot_mask, (soft_mask_ksize, soft_mask_ksize), 0)
        sum_slot_pixels = float(local_slot_mask_for_matching.sum())
    else:
        local_slot_mask_for_matching = local_slot_mask
        sum_slot_pixels = int(local_slot_mask.sum() // 255)

    piece_moments = cv2.moments((piece_crop > 0).astype(np.uint8))
    piece_cx, piece_cy = (piece_moments["m10"] / piece_moments["m00"], piece_moments["m01"] / piece_moments["m00"]) if piece_moments["m00"] else (piece_crop.shape[1]/2.0, piece_crop.shape[0]/2.0)
    
    p_h, p_w = piece_crop.shape[:2]
    pad_size = int(max(p_h, p_w) * 1.5)
    piece_centered_canvas = np.zeros((pad_size, pad_size), dtype=np.uint8)
    pcx, pcy = pad_size / 2.0, pad_size / 2.0
    
    px0 = int(round(pcx - piece_cx))
    py0 = int(round(pcy - piece_cy))
    piece_centered_canvas[py0:py0+p_h, px0:px0+p_w] = piece_crop

    def evaluate_fast(variant_centered_canvas, angle, mirror_mode, dx_list, dy_list):
        nonlocal best
        rot_matrix = cv2.getRotationMatrix2D((pcx, pcy), angle, 1.0)
        rotated = cv2.warpAffine(variant_centered_canvas, rot_matrix, (pad_size, pad_size), flags=cv2.INTER_NEAREST, borderValue=0)
        placed_rotated = place_mask_local_no_moments(rotated, local_shape, local_center[0], local_center[1], pcx, pcy)
        
        if use_soft_iou and soft_mask_ksize > 1:
            placed_rotated_blurred = cv2.GaussianBlur(placed_rotated, (soft_mask_ksize, soft_mask_ksize), 0)
            sum_rotated_pixels = float(placed_rotated_blurred.sum())
        else:
            placed_rotated_blurred = placed_rotated
            sum_rotated_pixels = int(placed_rotated.sum() // 255)

        if sum_rotated_pixels == 0 or sum_slot_pixels == 0:
            return 0.0

        H, W = local_shape
        max_iou_for_angle = 0.0
        
        for dx in dx_list:
            for dy in dy_list:
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

                if use_soft_iou and soft_mask_ksize > 1:
                    intersection_sum = np.minimum(
                        placed_rotated_blurred[ay0:ay1, ax0:ax1],
                        local_slot_mask_for_matching[by0:by1, bx0:bx1]
                    ).sum()
                    union_sum = sum_rotated_pixels + sum_slot_pixels - intersection_sum
                    iou = float(intersection_sum / union_sum) if union_sum > 0 else 0.0
                else:
                    intersection_sum = (placed_rotated_blurred[ay0:ay1, ax0:ax1] & local_slot_mask_for_matching[by0:by1, bx0:bx1]).sum()
                    intersection_pixels = int(intersection_sum // 255)
                    union_pixels = sum_rotated_pixels + sum_slot_pixels - intersection_pixels
                    iou = intersection_pixels / union_pixels if union_pixels > 0 else 0.0

                if iou > max_iou_for_angle:
                    max_iou_for_angle = iou
                    
                if iou > best["iou"]:
                    best = {
                        "iou": iou,
                        "rotation": int(angle % 360),
                        "dx": int(dx),
                        "dy": int(dy),
                        "mirrored": mirror_mode != "none",
                        "mirror_mode": mirror_mode,
                        "transform": {
                            "slot_center": [float(slot_center[0]), float(slot_center[1])],
                            "rotated_shape": [int(rotated.shape[1]), int(rotated.shape[0])],
                        },
                    }
        return max_iou_for_angle

    coarse_candidates = []
    for mirror_mode, variant_crop in mirror_variants(piece_crop, allow_mirror):
        v_moments = cv2.moments((variant_crop > 0).astype(np.uint8))
        v_cx, v_cy = (v_moments["m10"] / v_moments["m00"], v_moments["m01"] / v_moments["m00"]) if v_moments["m00"] else (variant_crop.shape[1]/2.0, variant_crop.shape[0]/2.0)
        v_canvas = np.zeros((pad_size, pad_size), dtype=np.uint8)
        v_x0 = int(round(pcx - v_cx))
        v_y0 = int(round(pcy - v_cy))
        v_canvas[v_y0:v_y0+variant_crop.shape[0], v_x0:v_x0+variant_crop.shape[1]] = variant_crop
        
        for angle in range(0, 360, coarse_step):
            iou = evaluate_fast(v_canvas, angle, mirror_mode, [0], [0])
            coarse_candidates.append((iou, angle, mirror_mode, v_canvas))

    coarse_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = coarse_candidates[:3]  # Only top 3 angles for refinement!

    for _, best_coarse_angle, mirror_mode, v_canvas in top_candidates:
        refine_angles = range(int(best_coarse_angle) - refine_range, int(best_coarse_angle) + refine_range + 1, max(1, fine_step))
        for angle in refine_angles:
            evaluate_fast(v_canvas, angle % 360, mirror_mode, translations, translations)

    return best


# Let's run matching for ALL 4 pieces over their candidates
print("=== BENCHMARKING COMPOSITE SYSTEM (disk cached) ===")

t_orig_start = time.time()
orig_results = {}
for p_idx, piece in enumerate(pieces):
    image_size = tuple(board_config["rectified_size"])
    piece_mask_full = contour_to_mask(piece["contour"], image_size)
    candidates = find_candidate_slots(piece, board_config["slots"], cfg)
    
    # Run original matching with cached masks
    best_iou = -1
    best_slot = None
    for slot in candidates:
        slot_mask = get_cached_slot_mask(board_id, slot)
        res = original_match(piece_mask_full, slot_mask, cfg)
        if res["iou"] > best_iou:
            best_iou = res["iou"]
            best_slot = slot["slot_id"]
    orig_results[piece["piece_id"]] = (best_slot, best_iou)
t_orig_end = time.time()
print(f"Original with Cached Masks: {t_orig_end - t_orig_start:.4f}s")
print(orig_results)

t_opt_start = time.time()
opt_results = {}
for p_idx, piece in enumerate(pieces):
    image_size = tuple(board_config["rectified_size"])
    piece_mask_full = contour_to_mask(piece["contour"], image_size)
    candidates = find_candidate_slots(piece, board_config["slots"], cfg)
    
    # Run optimized matching with cached masks
    best_iou = -1
    best_slot = None
    for slot in candidates:
        slot_mask = get_cached_slot_mask(board_id, slot)
        res = optimized_match(piece_mask_full, slot_mask, cfg)
        if res["iou"] > best_iou:
            best_iou = res["iou"]
            best_slot = slot["slot_id"]
    opt_results[piece["piece_id"]] = (best_slot, best_iou)
t_opt_end = time.time()
print(f"\nOptimized with Cached Masks: {t_opt_end - t_opt_start:.4f}s")
print(opt_results)

speedup = (t_orig_end - t_orig_start) / (t_opt_end - t_opt_start)
print(f"\nSpeedup: {speedup:.2f}x!")
