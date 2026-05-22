from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .board_builder import contour_features, contour_from_mask
from .calibration import order_corners, rectify_board
from .config import BOARDS_DIR
from .io_utils import write_json


def find_board_corners(image: np.ndarray, dark_threshold: int = 235) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    # The photographed board is black/gray with low saturation, while the
    # tabletop is bright pink with high saturation. This mask is more reliable
    # than grayscale alone because pink can be dark enough to pass a gray
    # threshold.
    mask = np.where((saturation < 95) & (value < dark_threshold), 255, 0).astype(np.uint8)
    kernel = np.ones((35, 35), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Unable to find board contour")

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(corners)


def detect_white_grid_lines(
    rectified: np.ndarray,
    white_threshold: int = 150,
    adaptive_block_size: int = 31,
    adaptive_c: int = -8,
    line_mode: str = "tophat_hsv",
    tophat_kernel_size: int = 17,
    hsv_s_max: int = 120,
    hsv_v_min: int = 100,
    component_min_area: int = 5,
    component_max_area: int = 800,
    component_min_ratio: float = 1.0,
    debug_images: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    if line_mode == "simple_binary":
        _, line_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        line_mask = filter_line_components(
            line_mask,
            min_area=component_min_area,
            max_area=component_max_area,
            min_ratio=component_min_ratio,
        )
        if debug_images is not None:
            debug_images["simple_binary_threshold.png"] = line_mask
            debug_images["line_mask_filtered.png"] = line_mask
        return line_mask

    if line_mode in {"tophat_hsv", "hybrid"}:
        kernel_size = max(3, tophat_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, th = cv2.threshold(tophat, white_threshold, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, hsv_v_min), (180, hsv_s_max, 255))
        raw_mask = cv2.bitwise_and(th, white_mask)
        line_mask = filter_line_components(
            raw_mask,
            min_area=component_min_area,
            max_area=component_max_area,
            min_ratio=component_min_ratio,
        )

        if line_mode == "hybrid":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            background = cv2.medianBlur(enhanced, 31)
            normalized = cv2.subtract(enhanced, background)
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
            block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
            adaptive_mask = cv2.adaptiveThreshold(
                normalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                max(block_size, 3),
                adaptive_c,
            )
            adaptive_mask = cv2.bitwise_and(adaptive_mask, white_mask)
            adaptive_mask = filter_line_components(
                adaptive_mask,
                min_area=component_min_area,
                max_area=component_max_area * 4 if component_max_area > 0 else 0,
                min_ratio=component_min_ratio,
            )
            line_mask = cv2.bitwise_or(line_mask, adaptive_mask)
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
            if debug_images is not None:
                debug_images["hybrid_adaptive_mask.png"] = adaptive_mask

        if debug_images is not None:
            debug_images["tophat.png"] = tophat
            debug_images["tophat_threshold.png"] = th
            debug_images["hsv_white_mask.png"] = white_mask
            debug_images["line_mask_raw.png"] = raw_mask
            debug_images["line_mask_filtered.png"] = line_mask
        return line_mask

    if line_mode == "adaptive":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        background = cv2.medianBlur(enhanced, 31)
        normalized = cv2.subtract(enhanced, background)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

        _, global_mask = cv2.threshold(normalized, white_threshold, 255, cv2.THRESH_BINARY)

        block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        adaptive_mask = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            max(block_size, 3),
            adaptive_c,
        )

        line_mask = cv2.bitwise_and(global_mask, adaptive_mask)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        if debug_images is not None:
            debug_images["normalized.png"] = normalized
            debug_images["adaptive_mask.png"] = adaptive_mask
            debug_images["line_mask_filtered.png"] = line_mask
        return line_mask

    raise ValueError(f"Unsupported line detection mode: {line_mode}")


def filter_line_components(
    line_mask: np.ndarray,
    min_area: int = 5,
    max_area: int = 800,
    min_ratio: float = 1.0,
) -> np.ndarray:
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, 8)
    clean = np.zeros_like(line_mask)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        ratio = max(width, height) / max(1, min(width, height))

        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if ratio < min_ratio:
            continue

        clean[labels == label] = 255

    return clean


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, opened)
        eroded = cv2.erode(binary, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded
        if cv2.countNonZero(binary) == 0:
            break

    return skeleton


def find_skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    import cv2

    binary = (skeleton > 0).astype(np.uint8)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((binary == 1) & (neighbor_count == 2))
    return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))


def connect_skeleton_endpoints(
    skeleton: np.ndarray,
    gap_limit: int = 8,
    max_connections_per_endpoint: int = 1,
) -> np.ndarray:
    import cv2

    if gap_limit <= 0:
        return skeleton

    output = skeleton.copy()
    endpoints = find_skeleton_endpoints(output)
    used: dict[tuple[int, int], int] = {}

    for index, p1 in enumerate(endpoints):
        if used.get(p1, 0) >= max_connections_per_endpoint:
            continue

        best: tuple[float, tuple[int, int]] | None = None
        for p2 in endpoints[index + 1 :]:
            if used.get(p2, 0) >= max_connections_per_endpoint:
                continue
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist == 0 or dist > gap_limit:
                continue
            if best is None or dist < best[0]:
                best = (dist, p2)

        if best is None:
            continue

        p2 = best[1]
        cv2.line(output, p1, p2, 255, 1)
        used[p1] = used.get(p1, 0) + 1
        used[p2] = used.get(p2, 0) + 1

    return output


def close_skeleton(skeleton: np.ndarray, close_iterations: int = 1) -> np.ndarray:
    import cv2

    if close_iterations <= 0:
        return skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)


def components_to_slots(
    line_mask: np.ndarray,
    min_slot_area: int = 1000,
    max_slot_area: int | None = None,
    border_thickness: int = 8,
    wall_dilate: int = 2,
    rows: int | None = None,
    cols: int | None = None,
) -> list[np.ndarray]:
    import cv2

    height, width = line_mask.shape[:2]
    if wall_dilate > 0:
        barriers = cv2.dilate(line_mask, np.ones((wall_dilate, wall_dilate), dtype=np.uint8), iterations=1)
    else:
        barriers = line_mask.copy()
    cv2.rectangle(barriers, (0, 0), (width - 1, height - 1), 255, border_thickness)

    free_space = cv2.bitwise_not(barriers)

    # Find seeds
    seeds: list[tuple[int, int]] = []
    if rows is not None and cols is not None and rows > 0 and cols > 0:
        # 1. Grid-based seeds
        cell_w = width / cols
        cell_h = height / rows

        def find_nearest_free_local(mask: np.ndarray, x: int, y: int, max_radius: int = 40) -> tuple[int, int] | None:
            h_m, w_m = mask.shape
            if 0 <= x < w_m and 0 <= y < h_m and mask[y, x] > 0:
                return x, y
            for r in range(1, max_radius + 1):
                x1 = max(0, x - r)
                x2 = min(w_m - 1, x + r)
                y1 = max(0, y - r)
                y2 = min(h_m - 1, y + r)
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        if mask[yy, xx] > 0:
                            return xx, yy
            return None

        for r in range(rows):
            for c in range(cols):
                cx = int((c + 0.5) * cell_w)
                cy = int((r + 0.5) * cell_h)
                found = find_nearest_free_local(free_space, cx, cy, max_radius=int(min(cell_w, cell_h) * 0.4))
                if found is not None:
                    seeds.append(found)
    else:
        # 2. Auto-detect seeds using distance transform local maxima
        dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
        max_dist = np.max(dist)
        if max_dist > 5:
            # Self-scaling window size for local maxima
            min_dist = max(15, int(max_dist * 0.6))
            kernel_size = min_dist if min_dist % 2 == 1 else min_dist + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
            local_max = cv2.dilate(dist_smooth, kernel)

            # Peaks: local max and distance greater than a small threshold
            peaks_mask = (dist_smooth == local_max) & (dist_smooth > 5)
            
            # Group adjacent peak pixels to avoid multiple seeds on plateaus
            num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
            
            # Get the centroids of each unique peak component
            seeds_list = []
            for label_id in range(1, num_peaks):
                cx, cy = peak_centroids[label_id]
                px = int(round(cx))
                py = int(round(cy))
                # Get the distance value at the centroid (clipped to bounds)
                py_clamped = min(max(0, py), height - 1)
                px_clamped = min(max(0, px), width - 1)
                d_val = float(dist_smooth[py_clamped, px_clamped])
                seeds_list.append((px, py, d_val))
                
            # Sort seeds by distance descending to prioritize central/larger areas
            seeds_with_dist = sorted(seeds_list, key=lambda item: item[2], reverse=True)
            seeds = [(x, y) for x, y, _ in seeds_with_dist]

    if not seeds:
        # Fallback to connected components if no seeds are found
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=8)
        if max_slot_area is None:
            max_slot_area = int(width * height * 0.02)
        slots: list[np.ndarray] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_border = x <= border_thickness or y <= border_thickness or x + w >= width - border_thickness or y + h >= height - border_thickness
            if touches_border:
                continue
            if area < min_slot_area or (max_slot_area is not None and area > max_slot_area):
                continue
            mask = np.zeros_like(line_mask)
            mask[labels == label] = 255
            slots.append(mask)
        return slots

    # Set up watershed markers
    markers = np.zeros(free_space.shape, dtype=np.int32)
    for label_id, (x, y) in enumerate(seeds, start=1):
        markers[y, x] = label_id

    # Run watershed
    # To run watershed, we need a 3-channel image. We use the negated free_space.
    gray_for_watershed = cv2.bitwise_not(free_space)
    bgr_for_watershed = cv2.cvtColor(gray_for_watershed, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr_for_watershed, markers)

    # Extract slots from watershed labels
    if max_slot_area is None:
        max_slot_area = int(width * height * 0.02)

    slots: list[np.ndarray] = []
    for label_id in range(1, len(seeds) + 1):
        mask = (markers == label_id) & (free_space > 0)
        area = np.count_nonzero(mask)
        if area < min_slot_area or (max_slot_area is not None and area > max_slot_area):
            continue

        # Check if the mask touches the border
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        touches_border = x1 <= border_thickness or y1 <= border_thickness or x2 >= width - border_thickness or y2 >= height - border_thickness
        if touches_border:
            continue

        slot_mask = (mask.astype(np.uint8) * 255)
        slots.append(slot_mask)

    return slots


def sort_slot_masks(slot_masks: list[np.ndarray]) -> list[np.ndarray]:
    import cv2

    def key(mask: np.ndarray) -> tuple[int, int]:
        contour = contour_from_mask(mask)
        features = contour_features(contour)
        cx, cy = features["center"]
        # Bucket by y so a slightly wavy row still sorts left-to-right.
        return (int(cy // 40), int(cx))

    return sorted(slot_masks, key=key)


def build_board_from_slot_masks(
    board_id: str,
    rectified_size: tuple[int, int],
    slot_masks: list[np.ndarray],
    output_root: Path = BOARDS_DIR,
) -> dict[str, Any]:
    import cv2

    board_dir = output_root / board_id
    slots_dir = board_dir / "slots"
    slots_dir.mkdir(parents=True, exist_ok=True)

    slots: list[dict[str, Any]] = []
    for index, mask in enumerate(sort_slot_masks(slot_masks), start=1):
        slot_id = f"slot_{index:03d}"
        mask_path = slots_dir / f"{slot_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        contour = contour_from_mask(mask)
        slots.append(
            {
                "slot_id": slot_id,
                "mask_path": str(mask_path.relative_to(board_dir)).replace("\\", "/"),
                "contour": contour.reshape(-1, 2).astype(int).tolist(),
                "rotation_mode": "any",
                "enabled": True,
                **contour_features(contour),
            }
        )

    board_config = {
        "board_id": board_id,
        "version": 1,
        "rectified_size": [rectified_size[0], rectified_size[1]],
        "marker_type": "auto_photo",
        "slots": slots,
    }
    write_json(board_dir / "board_config.json", board_config)
    return board_config


def clean_binary_grid_mask(binary_mask: np.ndarray, min_area: int = 1000) -> np.ndarray:
    import cv2
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    clean = np.zeros_like(binary_mask)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == label] = 255
    return clean


def import_board_from_photo(
    image_path: str | Path,
    board_id: str,
    rectified_size: tuple[int, int],
    dark_threshold: int = 235,
    white_threshold: int = 150,
    adaptive_block_size: int = 31,
    adaptive_c: int = -8,
    line_mode: str = "tophat_hsv",
    tophat_kernel_size: int = 17,
    hsv_s_max: int = 120,
    hsv_v_min: int = 100,
    component_min_area: int = 5,
    component_max_area: int = 800,
    component_min_ratio: float = 1.0,
    gap_limit: int = 8,
    skeleton_close_iterations: int = 1,
    wall_dilate: int = 2,
    use_skeleton: bool = True,
    min_slot_area: int = 1000,
    max_slot_area: int | None = None,
    border_thickness: int = 8,
    debug: bool = False,
    output_root: Path = BOARDS_DIR,
    rows: int | None = None,
    cols: int | None = None,
    binary: bool | None = None,
) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # Auto-detect binary image if not explicitly set
    if binary is None:
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        num_binary_pixels = np.sum((gray_orig < 5) | (gray_orig > 250))
        binary = (num_binary_pixels / gray_orig.size) > 0.99

    corners = find_board_corners(image, dark_threshold=dark_threshold)
    rectified, matrix = rectify_board(image, corners.tolist(), rectified_size)
    debug_images: dict[str, np.ndarray] = {}

    if binary:
        # If binary, threshold to clean up rectified bilinear gray pixels
        gray_rect = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        _, raw_mask = cv2.threshold(gray_rect, 127, 255, cv2.THRESH_BINARY)
        # Filter out small noise speckles to isolate the main continuous grid
        line_mask = clean_binary_grid_mask(raw_mask, min_area=1000)
        use_skeleton_mode = False
    else:
        line_mask = detect_white_grid_lines(
            rectified,
            white_threshold=white_threshold,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            line_mode=line_mode,
            tophat_kernel_size=tophat_kernel_size,
            hsv_s_max=hsv_s_max,
            hsv_v_min=hsv_v_min,
            component_min_area=component_min_area,
            component_max_area=component_max_area,
            component_min_ratio=component_min_ratio,
            debug_images=debug_images,
        )
        use_skeleton_mode = use_skeleton

    if use_skeleton_mode:
        skeleton = skeletonize_mask(line_mask)
        connected_skeleton = connect_skeleton_endpoints(skeleton, gap_limit=gap_limit)
        closed_skeleton = close_skeleton(connected_skeleton, close_iterations=skeleton_close_iterations)
        region_lines = closed_skeleton
    else:
        skeleton = line_mask.copy()
        connected_skeleton = line_mask.copy()
        closed_skeleton = line_mask.copy()
        region_lines = line_mask

    slot_masks = components_to_slots(
        region_lines,
        min_slot_area=min_slot_area,
        max_slot_area=max_slot_area,
        border_thickness=border_thickness,
        wall_dilate=wall_dilate,
        rows=rows,
        cols=cols,
    )
    board_config = build_board_from_slot_masks(board_id, rectified_size, slot_masks, output_root)

    board_config["source_image"] = str(image_path)
    board_config["detected_corners"] = corners.tolist()
    board_config["transform_matrix"] = matrix.tolist()
    write_json(output_root / board_id / "board_config.json", board_config)

    if debug:
        debug_dir = output_root / board_id / "debug_import"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "rectified.png"), rectified)
        cv2.imwrite(str(debug_dir / "line_mask.png"), line_mask)
        for name, image in debug_images.items():
            cv2.imwrite(str(debug_dir / name), image)
        cv2.imwrite(str(debug_dir / "skeleton.png"), skeleton)
        cv2.imwrite(str(debug_dir / "connected_skeleton.png"), connected_skeleton)
        cv2.imwrite(str(debug_dir / "closed_skeleton.png"), closed_skeleton)

        overlay = rectified.copy()
        for slot in board_config["slots"]:
            contour = np.asarray(slot["contour"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
            x, y, _, _ = slot["bbox"]
            cv2.putText(overlay, slot["slot_id"], (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "slots_overlay.png"), overlay)
        board_config["debug"] = {
            "rectified_image_path": str(debug_dir / "rectified.png"),
            "line_mask_path": str(debug_dir / "line_mask.png"),
            "skeleton_path": str(debug_dir / "skeleton.png"),
            "connected_skeleton_path": str(debug_dir / "connected_skeleton.png"),
            "closed_skeleton_path": str(debug_dir / "closed_skeleton.png"),
            "slots_overlay_path": str(debug_dir / "slots_overlay.png"),
        }

    return board_config
