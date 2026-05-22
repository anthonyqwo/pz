from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import BOARDS_DIR, MatcherConfig


def contour_array(points: list[list[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.int32)
    return arr.reshape(-1, 1, 2)


def bbox_ratio_diff(piece: dict[str, Any], slot: dict[str, Any]) -> float:
    _, _, piece_w, piece_h = piece.get("bbox", [0, 0, 0, 0])
    _, _, slot_w, slot_h = slot.get("bbox", [0, 0, 0, 0])
    piece_ratio = float(piece_w / piece_h) if piece_h else 0.0
    slot_ratio = float(slot_w / slot_h) if slot_h else 0.0
    if piece_ratio <= 0.0 or slot_ratio <= 0.0:
        return 1.0
    normal = abs(piece_ratio - slot_ratio) / max(slot_ratio, 0.001)
    inverted_slot = 1.0 / max(slot_ratio, 0.001)
    inverted = abs(piece_ratio - inverted_slot) / max(inverted_slot, 0.001)
    return float(min(normal, inverted))


def shape_score_to_similarity(shape_score: float | None) -> float:
    if shape_score is None:
        return 0.0
    return float(1.0 / (1.0 + max(float(shape_score), 0.0)))


def feature_distance_ok(piece: dict[str, Any], slot: dict[str, Any], config: MatcherConfig) -> bool:
    slot_area = max(float(slot["area"]), 1.0)
    area_diff = abs(float(piece["area"]) - slot_area) / slot_area
    return area_diff <= config.area_tolerance


def candidate_prefilter_score(area_diff: float, bbox_diff: float, shape_score: float) -> float:
    shape_penalty = max(shape_score, 0.0) / (1.0 + max(shape_score, 0.0))
    return float(0.50 * area_diff + 0.25 * min(bbox_diff, 2.0) + 0.25 * shape_penalty)


def find_candidate_slots(
    piece: dict[str, Any],
    slots: list[dict[str, Any]],
    config: MatcherConfig | None = None,
    max_candidates: int | None = None,
    relaxed: bool = True,
) -> list[dict[str, Any]]:
    import cv2

    cfg = config or MatcherConfig()
    limit = max_candidates or cfg.max_candidates
    piece_contour = contour_array(piece["contour"])
    scored_slots: list[dict[str, Any]] = []

    for slot in slots:
        if not slot.get("enabled", True):
            continue

        slot_area = max(float(slot["area"]), 1.0)
        area_diff = abs(float(piece["area"]) - slot_area) / slot_area
        bbox_diff = bbox_ratio_diff(piece, slot)
        slot_contour = contour_array(slot["contour"])
        shape_score = float(cv2.matchShapes(piece_contour, slot_contour, cv2.CONTOURS_MATCH_I1, 0))
        debug_flags: list[str] = []
        if area_diff > cfg.area_tolerance:
            debug_flags.append("area_outside_relaxed_tolerance")
        if bbox_diff > cfg.aspect_ratio_tolerance:
            debug_flags.append("bbox_ratio_penalty")

        scored_slots.append(
            {
                **slot,
                "area_diff_ratio": float(area_diff),
                "bbox_ratio_diff": float(bbox_diff),
                "shape_score": shape_score,
                "prefilter_score": candidate_prefilter_score(area_diff, bbox_diff, shape_score),
                "reason": "kept_by_relaxed_prefilter",
                "debug_flags": debug_flags,
            }
        )

    if not scored_slots:
        return []

    within_area = [item for item in scored_slots if item["area_diff_ratio"] <= cfg.area_tolerance]
    fallback_used = False
    if relaxed and len(within_area) < cfg.min_candidates_before_relax:
        candidates = scored_slots
        fallback_used = True
        fallback_flag = "relaxed_min_candidates"
    elif within_area:
        candidates = within_area
        fallback_flag = ""
    else:
        candidates = scored_slots
        fallback_used = True
        fallback_flag = "fallback_all_slots"

    candidates.sort(key=lambda item: item["prefilter_score"])
    kept = candidates[: max(1, limit)]
    if fallback_used:
        for item in kept:
            item["reason"] = fallback_flag
            item["debug_flags"] = [*item.get("debug_flags", []), fallback_flag]
    return kept


def crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("mask has no contour")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return mask[y : y + h, x : x + w], (x, y, w, h)


def contour_to_mask(contour_points: list[list[int]], image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    contour = contour_array(contour_points)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    import cv2

    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"]:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    h, w = mask.shape[:2]
    return float(w / 2.0), float(h / 2.0)


def rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]
    rotated = cv2.warpAffine(mask, matrix, (new_width, new_height), flags=cv2.INTER_NEAREST, borderValue=0)
    _, binary = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
    return binary


def place_centered(mask: np.ndarray, canvas_shape: tuple[int, int], center: list[float] | tuple[float, float]) -> np.ndarray:
    return place_by_centroid(mask, canvas_shape, center)


def place_by_centroid(
    mask: np.ndarray,
    canvas_shape: tuple[int, int],
    target_center: list[float] | tuple[float, float],
    dx: int = 0,
    dy: int = 0,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    h, w = mask.shape[:2]
    cx, cy = mask_centroid(mask)
    target_x, target_y = target_center
    x0 = int(round(float(target_x) + dx - cx))
    y0 = int(round(float(target_y) + dy - cy))
    x1 = x0 + w
    y1 = y0 + h

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(canvas_w, x1)
    dst_y1 = min(canvas_h, y1)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return result

    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return result


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a > 0
    b_bool = b > 0
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def load_slot_mask(board_id: str, slot: dict[str, Any], board_root: Path = BOARDS_DIR) -> np.ndarray:
    import cv2

    path = board_root / board_id / slot["mask_path"]
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read slot mask: {path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def angle_values(start: int, stop: int, step: int) -> list[int]:
    step = max(1, int(step))
    return list(range(start, stop, step))


def mirror_variants(mask: np.ndarray, allow_mirror: bool) -> list[tuple[str, np.ndarray]]:
    variants = [("none", mask)]
    if allow_mirror:
        variants.extend([("mirror_x", np.fliplr(mask)), ("mirror_y", np.flipud(mask))])
    return variants


def match_piece_to_slot_iou(
    piece_mask: np.ndarray,
    slot_mask: np.ndarray,
    options: MatcherConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = options or MatcherConfig()
    get = cfg.get if isinstance(cfg, dict) else lambda key, default=None: getattr(cfg, key, default)
    coarse_step = int(get("angle_step_coarse", 5))
    refine_range = int(get("angle_refine_range", 5))
    fine_step = int(get("angle_step_fine", 1))
    translations = list(get("translation_search", [-10, -5, 0, 5, 10]))
    allow_mirror = bool(get("allow_mirror", False))

    piece_crop, _ = crop_mask(piece_mask)
    slot_center = mask_centroid(slot_mask)
    best: dict[str, Any] = {
        "iou": 0.0,
        "rotation": 0,
        "dx": 0,
        "dy": 0,
        "mirrored": False,
        "mirror_mode": "none",
        "transform": {},
    }

    def evaluate(variant_mask: np.ndarray, angle: int, mirror_mode: str) -> None:
        nonlocal best
        rotated = rotate_mask(variant_mask, angle)
        for dx in translations:
            for dy in translations:
                placed = place_by_centroid(rotated, slot_mask.shape, slot_center, int(dx), int(dy))
                iou = mask_iou(placed, slot_mask)
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
        for angle in angle_values(0, 360, coarse_step):
            evaluate(variant_mask, angle, mirror_mode)

    refine_angles = range(int(best["rotation"]) - refine_range, int(best["rotation"]) + refine_range + 1, max(1, fine_step))
    for mirror_mode, variant_mask in mirror_variants(piece_crop, allow_mirror):
        for angle in refine_angles:
            evaluate(variant_mask, angle % 360, mirror_mode)

    return best


def confidence_scores(best: dict[str, Any], second_iou: float, config: MatcherConfig) -> dict[str, float]:
    iou_score = float(best.get("iou", 0.0))
    area_score = max(0.0, 1.0 - float(best.get("area_diff_ratio", 1.0)))
    shape_score_normalized = shape_score_to_similarity(best.get("shape_score"))
    margin_score = min(max((iou_score - float(second_iou)) / max(config.min_margin, 0.001), 0.0), 1.0)
    confidence = (
        config.confidence_iou_weight * iou_score
        + config.confidence_area_weight * area_score
        + config.confidence_shape_weight * shape_score_normalized
        + config.confidence_margin_weight * margin_score
    )
    return {
        "confidence": float(confidence),
        "iou_score": iou_score,
        "area_score": float(area_score),
        "shape_score_normalized": float(shape_score_normalized),
        "margin_score": float(margin_score),
    }


def score_piece_slot(
    piece: dict[str, Any],
    slot: dict[str, Any],
    board_id: str,
    image_size: tuple[int, int],
    config: MatcherConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MatcherConfig()
    slot_mask = load_slot_mask(board_id, slot)
    piece_mask_full = contour_to_mask(piece["contour"], image_size)
    iou_result = match_piece_to_slot_iou(piece_mask_full, slot_mask, cfg)

    return {
        "slot_id": slot["slot_id"],
        "iou": iou_result["iou"],
        "rotation": iou_result["rotation"],
        "dx": iou_result["dx"],
        "dy": iou_result["dy"],
        "mirrored": iou_result["mirrored"],
        "mirror_mode": iou_result["mirror_mode"],
        "transform": iou_result["transform"],
        "shape_score": slot.get("shape_score"),
        "prefilter_score": slot.get("prefilter_score"),
        "bbox_ratio_diff": slot.get("bbox_ratio_diff"),
        "area_diff_ratio": slot.get(
            "area_diff_ratio",
            abs(float(piece["area"]) - float(slot["area"])) / max(float(slot["area"]), 1.0),
        ),
        "debug_flags": slot.get("debug_flags", []),
    }


def classify_status(best_iou: float, second_iou: float, config: MatcherConfig) -> str:
    iou_gap = best_iou - second_iou
    if best_iou >= config.confident_iou and iou_gap >= config.min_margin:
        return "confident"
    if best_iou >= config.ambiguous_iou:
        return "ambiguous"
    return "rejected"


def reject_reason_for(status: str, best_iou: float, second_iou: float, config: MatcherConfig) -> str | None:
    if status != "rejected":
        return None
    if best_iou >= config.confident_iou and best_iou - second_iou < config.min_margin:
        return "iou_margin_too_small"
    if best_iou < config.ambiguous_iou:
        return "best_iou_below_ambiguous_threshold"
    return "decision_threshold_not_met"


def diagnose_match(candidates: list[dict[str, Any]], scored: list[dict[str, Any]], status: str, config: MatcherConfig) -> str | None:
    if not candidates:
        return "candidate_filter_too_strict"
    if not scored:
        return "candidate_filter_too_strict"
    best = scored[0]
    second_iou = scored[1]["iou"] if len(scored) > 1 else 0.0
    if best["iou"] >= config.confident_iou and status == "rejected":
        return "decision_threshold_too_strict"
    if best["iou"] < config.ambiguous_iou:
        return "mask_quality_or_alignment_issue"
    if best["iou"] - second_iou < config.min_margin:
        return "ambiguous_similar_slots"
    if abs(int(best.get("dx", 0))) > 0 or abs(int(best.get("dy", 0))) > 0:
        return "center_alignment_error"
    return None


def match_piece_to_board(
    piece: dict[str, Any],
    board_config: dict[str, Any],
    config: MatcherConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MatcherConfig()
    image_size = tuple(board_config["rectified_size"])
    all_enabled_slots = [slot for slot in board_config["slots"] if slot.get("enabled", True)]
    candidates = find_candidate_slots(piece, board_config["slots"], cfg)
    fallback_used = any(
        flag in {"fallback_all_slots", "relaxed_min_candidates"}
        for candidate in candidates
        for flag in candidate.get("debug_flags", [])
    )

    if not candidates and all_enabled_slots:
        candidates = find_candidate_slots(piece, all_enabled_slots, cfg, max_candidates=len(all_enabled_slots), relaxed=True)
        fallback_used = True

    scored = [
        score_piece_slot(piece, slot, board_config["board_id"], image_size, cfg)
        for slot in candidates
    ]
    scored.sort(key=lambda item: item["iou"], reverse=True)

    if not scored:
        decision = {
            "best_slot_id": None,
            "best_iou": 0.0,
            "second_iou": 0.0,
            "iou_gap": 0.0,
            "status": "rejected",
            "reject_reason": "no_candidates",
        }
        return {
            "piece_id": piece["piece_id"],
            "matched_slot_id": None,
            "rotation": None,
            "dx": 0,
            "dy": 0,
            "mirrored": False,
            "mirror_mode": "none",
            "confidence": 0.0,
            "confidence_raw": {},
            "iou": 0.0,
            "status": "rejected",
            "diagnosis": "candidate_filter_too_strict",
            "candidates": [],
            "debug": build_debug(piece, all_enabled_slots, candidates, scored, fallback_used, decision),
        }

    best = scored[0]
    second_iou = scored[1]["iou"] if len(scored) > 1 else 0.0
    status = classify_status(best["iou"], second_iou, cfg)
    confidence = confidence_scores(best, second_iou, cfg)
    for item in scored:
        item_confidence = confidence_scores(item, second_iou if item is best else best["iou"], cfg)
        item["confidence"] = item_confidence["confidence"]
        item["confidence_raw"] = item_confidence

    decision = {
        "best_slot_id": best["slot_id"],
        "best_iou": best["iou"],
        "second_iou": second_iou,
        "iou_gap": best["iou"] - second_iou,
        "status": status,
        "reject_reason": reject_reason_for(status, best["iou"], second_iou, cfg),
    }
    diagnosis = diagnose_match(candidates, scored, status, cfg)

    return {
        "piece_id": piece["piece_id"],
        "matched_slot_id": best["slot_id"],
        "rotation": best["rotation"],
        "dx": best["dx"],
        "dy": best["dy"],
        "mirrored": best["mirrored"],
        "mirror_mode": best["mirror_mode"],
        "confidence": confidence["confidence"],
        "confidence_raw": confidence,
        "iou": best["iou"],
        "shape_score": best.get("shape_score"),
        "area_diff_ratio": best.get("area_diff_ratio"),
        "status": status,
        "diagnosis": diagnosis,
        "candidates": scored[:5],
        "debug": build_debug(piece, all_enabled_slots, candidates, scored, fallback_used, decision),
    }


def candidate_debug_item(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot_id": candidate.get("slot_id"),
        "area_diff_ratio": candidate.get("area_diff_ratio"),
        "bbox_ratio_diff": candidate.get("bbox_ratio_diff"),
        "shape_score": candidate.get("shape_score"),
        "prefilter_score": candidate.get("prefilter_score"),
        "reason": candidate.get("reason"),
        "debug_flags": candidate.get("debug_flags", []),
    }


def match_debug_item(match: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "slot_id",
        "iou",
        "rotation",
        "dx",
        "dy",
        "mirrored",
        "mirror_mode",
        "area_diff_ratio",
        "bbox_ratio_diff",
        "shape_score",
        "confidence",
        "confidence_raw",
    ]
    return {key: match.get(key) for key in keys if key in match}


def build_debug(
    piece: dict[str, Any],
    all_slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    fallback_used: bool,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "piece_id": piece.get("piece_id"),
        "detected_piece": {
            "area": piece.get("area"),
            "bbox": piece.get("bbox"),
            "center": piece.get("center"),
        },
        "candidate_filter": {
            "total_slots": len(all_slots),
            "kept_candidates": len(candidates),
            "fallback_used": bool(fallback_used),
            "top_candidates_before_iou": [candidate_debug_item(item) for item in candidates[:10]],
        },
        "matching": {
            "top_matches": [match_debug_item(item) for item in scored[:10]],
        },
        "decision": decision,
    }
