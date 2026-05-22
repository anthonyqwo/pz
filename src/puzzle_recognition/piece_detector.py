from __future__ import annotations

from typing import Any

import numpy as np

from .board_builder import contour_features
from .config import DetectorConfig


def lightness_channel(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]
    return image


def threshold_black(image: np.ndarray, gray_threshold: int, pre_blur_ksize: int = 5) -> np.ndarray:
    import cv2

    L = lightness_channel(image)
    if pre_blur_ksize > 1:
        ksize = int(pre_blur_ksize)
        if ksize % 2 == 0:
            ksize += 1
        L = cv2.GaussianBlur(L, (ksize, ksize), 0)
    _, mask = cv2.threshold(L, gray_threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def smooth_contour_gaussian(contour: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    if sigma <= 0.0:
        return contour

    from scipy.ndimage import gaussian_filter1d
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 4:
        return contour

    smoothed_x = gaussian_filter1d(points[:, 0], sigma=sigma, mode="wrap")
    smoothed_y = gaussian_filter1d(points[:, 1], sigma=sigma, mode="wrap")

    smoothed = np.stack([smoothed_x, smoothed_y], axis=1)
    return smoothed.reshape(-1, 1, 2).astype(np.int32)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    height, width = padded.shape[:2]
    flood = padded.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def clean_black_mask(mask: np.ndarray, kernel_size: int, close_iterations: int = 2) -> np.ndarray:
    import cv2

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, close_iterations))
    return fill_mask_holes(closed)


def component_touches_border(bbox: tuple[int, int, int, int], image_shape: tuple[int, int], margin: int) -> bool:
    x, y, w, h = bbox
    height, width = image_shape
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin


def component_features(
    component_id: int,
    contour: np.ndarray,
    component_mask: np.ndarray,
    lightness: np.ndarray,
    border_margin: int,
) -> dict[str, Any]:
    import cv2

    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h)
    aspect_ratio = float(w / h) if h else 0.0
    extent = float(area / bbox_area) if bbox_area else 0.0
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = float(area / hull_area) if hull_area else 0.0
    mean_L = float(cv2.mean(lightness, mask=component_mask)[0])
    touches_border = component_touches_border((x, y, w, h), lightness.shape[:2], border_margin)

    return {
        "id": int(component_id),
        "area": area,
        "bbox": [int(x), int(y), int(w), int(h)],
        "bbox_area": bbox_area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "hull_area": hull_area,
        "solidity": solidity,
        "mean_L": mean_L,
        "touches_border": bool(touches_border),
    }


def stats_component_features(
    component_id: int,
    stats: np.ndarray,
    mean_L: float,
    image_shape: tuple[int, int],
    border_margin: int,
) -> dict[str, Any]:
    import cv2

    x = int(stats[component_id, cv2.CC_STAT_LEFT])
    y = int(stats[component_id, cv2.CC_STAT_TOP])
    w = int(stats[component_id, cv2.CC_STAT_WIDTH])
    h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    area = float(stats[component_id, cv2.CC_STAT_AREA])
    bbox_area = float(w * h)
    aspect_ratio = float(w / h) if h else 0.0
    touches_border = component_touches_border((x, y, w, h), image_shape, border_margin)
    contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

    return {
        "id": int(component_id),
        "area": area,
        "bbox": [x, y, w, h],
        "bbox_area": bbox_area,
        "aspect_ratio": aspect_ratio,
        "extent": 0.0,
        "hull_area": 0.0,
        "solidity": 0.0,
        "mean_L": float(mean_L),
        "touches_border": bool(touches_border),
        "contour": contour.tolist(),
    }


def cheap_reject_reasons(features: dict[str, Any], cfg: DetectorConfig) -> list[str]:
    reasons: list[str] = []
    _, _, w, h = features["bbox"]

    if features["area"] < cfg.min_piece_area:
        reasons.append("area_too_small")
    if features["area"] > cfg.max_piece_area:
        reasons.append("area_too_large")
    if w < cfg.min_bbox_width or h < cfg.min_bbox_height:
        reasons.append("bbox_too_small")
    if w > cfg.max_bbox_width or h > cfg.max_bbox_height:
        reasons.append("bbox_too_large")
    if not (cfg.aspect_ratio_min <= features["aspect_ratio"] <= cfg.aspect_ratio_max):
        reasons.append("aspect_ratio_out_of_range")
    if features["mean_L"] > cfg.max_piece_mean_L:
        reasons.append("mean_L_too_high")
    if cfg.reject_border_components and features["touches_border"]:
        reasons.append("touches_border")

    return reasons


def reject_reasons(features: dict[str, Any], cfg: DetectorConfig) -> list[str]:
    reasons: list[str] = []
    _, _, w, h = features["bbox"]

    if features["area"] < cfg.min_piece_area:
        reasons.append("area_too_small")
    if features["area"] > cfg.max_piece_area:
        reasons.append("area_too_large")
    if w < cfg.min_bbox_width or h < cfg.min_bbox_height:
        reasons.append("bbox_too_small")
    if w > cfg.max_bbox_width or h > cfg.max_bbox_height:
        reasons.append("bbox_too_large")
    if not (cfg.aspect_ratio_min <= features["aspect_ratio"] <= cfg.aspect_ratio_max):
        reasons.append("aspect_ratio_out_of_range")
    if not (cfg.extent_min <= features["extent"] <= cfg.extent_max):
        reasons.append("extent_out_of_range")
    if not (cfg.solidity_min <= features["solidity"] <= cfg.solidity_max):
        reasons.append("solidity_out_of_range")
    if features["mean_L"] > cfg.max_piece_mean_L:
        reasons.append("mean_L_too_high")
    if cfg.reject_border_components and features["touches_border"]:
        reasons.append("touches_border")

    return reasons


def draw_component_overlay(
    image: np.ndarray,
    components: list[dict[str, Any]],
    mode: str = "all",
    label_min_area: float = 0.0,
) -> np.ndarray:
    import cv2

    output = image.copy()
    for component in components:
        rejected = bool(component.get("reject_reasons"))
        if mode == "kept" and rejected:
            continue
        if mode == "rejected" and not rejected:
            continue

        contour = np.asarray(component["contour"], dtype=np.int32).reshape(-1, 1, 2)
        color = (0, 190, 0) if not rejected else (0, 0, 255)
        cv2.drawContours(output, [contour], -1, color, 2)
        if rejected and component["area"] < label_min_area:
            continue
        x, y, _, _ = component["bbox"]
        if rejected:
            label = component["reject_reasons"][0]
        else:
            label = component.get("piece_id", f"component_{component['id']}")
        cv2.putText(output, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def diagnosis_for_counts(raw_components: int, kept_pieces: int, cfg: DetectorConfig) -> str | None:
    if kept_pieces == 0 and raw_components > 0:
        return "filter_too_strict"
    if kept_pieces > cfg.expected_max_pieces:
        return "filter_too_loose_or_shadow_detected"
    return None


def detect_black_pieces(rectified_image: np.ndarray, config: DetectorConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2

    cfg = config or DetectorConfig()
    lightness = lightness_channel(rectified_image)
    raw_mask = threshold_black(rectified_image, cfg.gray_threshold, cfg.pre_blur_ksize)
    cleaned_mask = clean_black_mask(raw_mask, cfg.morphology_kernel_size, cfg.close_iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, 8)
    label_counts = np.bincount(labels.ravel(), minlength=num_labels).astype(np.float64)
    lightness_sums = np.bincount(labels.ravel(), weights=lightness.ravel(), minlength=num_labels)
    mean_lightness = np.divide(lightness_sums, label_counts, out=np.zeros_like(lightness_sums), where=label_counts > 0)
    components: list[dict[str, Any]] = []
    pieces: list[dict[str, Any]] = []
    kept_mask = np.zeros_like(cleaned_mask)

    for component_id in range(1, num_labels):
        stats_features = stats_component_features(component_id, stats, mean_lightness[component_id], lightness.shape[:2], cfg.border_margin)
        cheap_reasons = cheap_reject_reasons(stats_features, cfg)
        if cheap_reasons:
            components.append({**stats_features, "reject_reasons": cheap_reasons})
            continue

        component_mask = np.zeros_like(cleaned_mask)
        component_mask[labels == component_id] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        features = component_features(component_id, contour, component_mask, lightness, cfg.border_margin)
        reasons = reject_reasons(features, cfg)
        component = {
            **features,
            "contour": contour.reshape(-1, 2).astype(int).tolist(),
            "reject_reasons": reasons,
        }

        if reasons:
            components.append(component)
            continue

        # Smooth dense contour first, then simplify to avoid shape collapse
        smoothed_contour = smooth_contour_gaussian(contour, cfg.contour_smooth_sigma)
        epsilon = cfg.contour_epsilon_ratio * cv2.arcLength(smoothed_contour, True)
        smoothed_approx = cv2.approxPolyDP(smoothed_contour, epsilon, True) if epsilon > 0 else smoothed_contour
        feature_values = contour_features(smoothed_approx)
        piece = {
            "piece_id": f"piece_{len(pieces) + 1:03d}",
            "contour": smoothed_approx.reshape(-1, 2).astype(int).tolist(),
            "solidity": features["solidity"],
            "aspect_ratio": features["aspect_ratio"],
            "extent": features["extent"],
            "mean_L": features["mean_L"],
            "component_id": component_id,
            "features": {
                **feature_values,
                "aspect_ratio": features["aspect_ratio"],
                "extent": features["extent"],
                "solidity": features["solidity"],
                "mean_L": features["mean_L"],
            },
            **feature_values,
        }
        pieces.append(piece)
        kept_mask[labels == component_id] = 255
        components.append({**component, "piece_id": piece["piece_id"]})

    pieces.sort(key=lambda item: item["area"], reverse=True)
    for index, piece in enumerate(pieces, start=1):
        piece["piece_id"] = f"piece_{index:03d}"

    kept_by_component = {piece["component_id"]: piece["piece_id"] for piece in pieces}
    for component in components:
        if not component["reject_reasons"] and component["id"] in kept_by_component:
            component["piece_id"] = kept_by_component[component["id"]]

    rejected_components = [
        {key: value for key, value in component.items() if key != "contour"}
        for component in components
        if component["reject_reasons"]
    ]
    kept_components = [
        {
            "piece_id": piece["piece_id"],
            "area": piece["area"],
            "bbox": piece["bbox"],
            "center": piece["center"],
            "features": piece["features"],
        }
        for piece in pieces
    ]
    raw_components = len(components)
    diagnosis = diagnosis_for_counts(raw_components, len(pieces), cfg)
    debug = {
        "raw_components": raw_components,
        "kept_pieces": len(pieces),
        "rejected_components": rejected_components,
        "kept_components": kept_components,
        "diagnosis": diagnosis,
    }

    return pieces, {
        "black_mask": raw_mask,
        "raw_dark_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "kept_mask": kept_mask,
        "all_components_overlay": draw_component_overlay(rectified_image, components, "all", cfg.debug_label_min_area),
        "rejected_components_overlay": draw_component_overlay(rectified_image, components, "rejected", cfg.debug_label_min_area),
        "kept_pieces_overlay": draw_component_overlay(rectified_image, components, "kept", cfg.debug_label_min_area),
        "debug": debug,
    }
