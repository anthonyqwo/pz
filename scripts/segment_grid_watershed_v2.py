"""
Grid-Prior Watershed Puzzle Segmentation v2
=============================================
Key improvements over v1:
- Skip skeletonization entirely - it loses information
- Use heavy morphological closing directly on binary mask
- Use cv2.inpaint for gap repair
- Better watershed: use distance transform + edge energy combined
- Two-pass: detect merged regions and re-split them
- Better border handling for edge pieces
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Step 1: Aggressive gap closing (no skeletonization)
# ---------------------------------------------------------------------------

def aggressive_gap_close(mask: np.ndarray, debug_dir: Path | None = None) -> np.ndarray:
    """
    Multi-strategy gap closing. Works on the thick binary lines directly.
    No skeletonization needed.
    """
    result = mask.copy()

    # Strategy 1: Standard morphological close with increasing kernel sizes
    for ksize in [3, 5, 7, 9]:
        k = np.ones((ksize, ksize), dtype=np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k, iterations=1)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01a_close_standard.png"), result)

    # Strategy 2: Directional closing (horizontal, vertical, diagonal)
    for length in [11, 17, 23]:
        for angle in [0, 45, 90, 135]:
            kernel = _line_kernel(length, angle)
            closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
            result = cv2.bitwise_or(result, closed)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01b_close_directional.png"), result)

    # Strategy 3: Dilate then erode (fills gaps then thins back)
    # Use a bigger dilation than erosion to ensure gaps are filled
    d_kernel = np.ones((5, 5), dtype=np.uint8)
    dilated = cv2.dilate(result, d_kernel, iterations=2)
    e_kernel = np.ones((3, 3), dtype=np.uint8)
    result = cv2.erode(dilated, e_kernel, iterations=2)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01c_dilate_erode.png"), result)

    return result


def _line_kernel(length: int, angle_deg: int) -> np.ndarray:
    """Create a line-shaped structuring element."""
    length = max(3, length)
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    angle_rad = np.deg2rad(angle_deg)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(angle_rad)))
        y = int(round(center - offset * np.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel


# ---------------------------------------------------------------------------
# Step 2: Generate grid seeds with proper alignment
# ---------------------------------------------------------------------------

def estimate_grid_offset(
    binary: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[float, float, float, float]:
    """
    Estimate the puzzle grid's offset and cell sizes by analyzing
    the projection profiles of the white lines.
    Returns (x_offset, y_offset, cell_w, cell_h).
    """
    height, width = binary.shape

    # Horizontal projection: sum white pixels per column
    h_proj = np.sum(binary > 0, axis=0).astype(float)
    # Vertical projection: sum white pixels per row
    v_proj = np.sum(binary > 0, axis=1).astype(float)

    # Smooth the projections
    from scipy.ndimage import uniform_filter1d
    h_smooth = uniform_filter1d(h_proj, size=5)
    v_smooth = uniform_filter1d(v_proj, size=5)

    # Find peaks in horizontal projection (vertical lines)
    h_peaks = _find_projection_peaks(h_smooth, cols + 1, width)
    v_peaks = _find_projection_peaks(v_smooth, rows + 1, height)

    if len(h_peaks) >= 2 and len(v_peaks) >= 2:
        cell_w = np.median(np.diff(sorted(h_peaks)))
        cell_h = np.median(np.diff(sorted(v_peaks)))
        x_off = h_peaks[0] if h_peaks[0] < cell_w else 0
        y_off = v_peaks[0] if v_peaks[0] < cell_h else 0
    else:
        cell_w = width / cols
        cell_h = height / rows
        x_off = cell_w / 2
        y_off = cell_h / 2

    return x_off, y_off, cell_w, cell_h


def _find_projection_peaks(projection: np.ndarray, expected_count: int, total_length: int) -> list[int]:
    """Find peaks in a 1D projection profile."""
    expected_spacing = total_length / (expected_count - 1) if expected_count > 1 else total_length

    # Find local maxima with minimum distance
    from scipy.signal import find_peaks
    min_dist = int(expected_spacing * 0.5)
    threshold = np.percentile(projection, 60)  # Only look at top 40%
    peaks, props = find_peaks(projection, distance=min_dist, height=threshold)

    return sorted(peaks.tolist())


def generate_grid_seeds(
    free_space: np.ndarray,
    rows: int,
    cols: int,
    binary_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """
    Generate seeds at estimated grid cell centers.
    Uses projection analysis if binary mask is provided, otherwise uniform grid.
    """
    height, width = free_space.shape

    # Try to estimate grid alignment
    if binary_mask is not None:
        x_off, y_off, cell_w, cell_h = estimate_grid_offset(binary_mask, rows, cols)
    else:
        cell_w = width / cols
        cell_h = height / rows
        x_off = cell_w / 2
        y_off = cell_h / 2

    print(f"  Grid: cell_w={cell_w:.1f}, cell_h={cell_h:.1f}, x_off={x_off:.1f}, y_off={y_off:.1f}")

    # Distance transform of free space for finding good seed positions
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)

    seeds = []
    search_radius = int(min(cell_w, cell_h) * 0.4)

    for r in range(rows):
        for c in range(cols):
            # Estimate cell center
            cx = int(x_off + c * cell_w)
            cy = int(y_off + r * cell_h)

            # Clamp
            cx = min(max(0, cx), width - 1)
            cy = min(max(0, cy), height - 1)

            # Find the position with maximum distance (furthest from walls)
            # within a search window around the estimated center
            seed = _find_best_seed(dist, cx, cy, search_radius)
            if seed is not None:
                seeds.append(seed)
            else:
                # Fallback: use center if in free space
                if free_space[cy, cx] > 0:
                    seeds.append((cx, cy))

    return seeds


def _find_best_seed(
    dist_map: np.ndarray,
    cx: int, cy: int,
    radius: int,
) -> tuple[int, int] | None:
    """Find the position with maximum distance transform value near (cx, cy)."""
    h, w = dist_map.shape
    y1 = max(0, cy - radius)
    y2 = min(h - 1, cy + radius)
    x1 = max(0, cx - radius)
    x2 = min(w - 1, cx + radius)

    sub = dist_map[y1:y2+1, x1:x2+1]
    if sub.max() <= 0:
        return None

    local_y, local_x = np.unravel_index(sub.argmax(), sub.shape)
    return (int(local_x) + x1, int(local_y) + y1)


# ---------------------------------------------------------------------------
# Step 3: Watershed segmentation
# ---------------------------------------------------------------------------

def run_watershed(
    free_space: np.ndarray,
    seeds: list[tuple[int, int]],
    wall_mask: np.ndarray,
) -> np.ndarray:
    """
    Marker-controlled watershed using wall mask as the gradient.
    """
    # Create markers
    markers = np.zeros(free_space.shape, dtype=np.int32)
    for label_id, (x, y) in enumerate(seeds, start=1):
        # Place a small circle marker, not just a point
        cv2.circle(markers, (x, y), 3, int(label_id), -1)

    # Create gradient image for watershed
    # Combine: distance transform (inverted) + wall proximity
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Walls should be ridges (high values), centers should be valleys
    gradient = cv2.bitwise_not(dist_norm)

    # Boost the wall edges: add the wall mask to make them stronger barriers
    wall_boost = (wall_mask > 0).astype(np.uint8) * 255
    gradient = cv2.addWeighted(gradient, 0.5, wall_boost, 0.5, 0)

    # Watershed needs BGR
    bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(bgr, markers)

    return labels


# ---------------------------------------------------------------------------
# Step 4: Extract and validate slots
# ---------------------------------------------------------------------------

def extract_and_validate_slots(
    labels: np.ndarray,
    free_space: np.ndarray,
    num_seeds: int,
    expected_area: float,
) -> tuple[list[dict], list[dict]]:
    """
    Extract slots from watershed labels.
    Returns (good_slots, problem_slots).
    """
    height, width = labels.shape
    min_area = int(expected_area * 0.25)
    max_area = int(expected_area * 2.0)
    border_margin = 10

    good_slots = []
    merged_slots = []
    fragment_slots = []

    for label_id in range(1, num_seeds + 1):
        region_mask = ((labels == label_id) & (free_space > 0)).astype(np.uint8) * 255
        area = cv2.countNonZero(region_mask)

        if area == 0:
            continue

        ys, xs = np.where(region_mask > 0)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        touches = (x_min <= border_margin or y_min <= border_margin or
                   x_max >= width - border_margin or y_max >= height - border_margin)

        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        # Aspect ratio check
        aspect = bbox_w / bbox_h if bbox_h > 0 else 0

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

        slot = {
            "label_id": label_id,
            "area": area,
            "contour_area": int(contour_area),
            "bbox": [x_min, y_min, bbox_w, bbox_h],
            "center": [cx, cy],
            "solidity": round(solidity, 3),
            "aspect_ratio": round(aspect, 3),
            "touches_border": touches,
            "contour": contour,
            "mask": region_mask,
        }

        # Classify
        if area > max_area and not touches:
            merged_slots.append(slot)
        elif area < min_area and not touches:
            fragment_slots.append(slot)
        else:
            good_slots.append(slot)

    return good_slots, merged_slots + fragment_slots


# ---------------------------------------------------------------------------
# Step 5: Re-split merged regions
# ---------------------------------------------------------------------------

def resplit_merged_regions(
    merged_slots: list[dict],
    free_space: np.ndarray,
    expected_area: float,
    wall_mask: np.ndarray,
) -> list[dict]:
    """
    Take merged regions (too large) and use distance-transform local maxima
    to re-split them into individual pieces.
    """
    new_slots = []

    for slot in merged_slots:
        mask = slot["mask"]
        area = slot["area"]

        # Estimate how many pieces are merged
        num_pieces = max(2, round(area / expected_area))

        # Distance transform on this region
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Find local maxima
        from skimage.feature import peak_local_max
        min_dist = int((expected_area ** 0.5) * 0.4)
        coords = peak_local_max(dist, min_distance=min_dist, num_peaks=num_pieces * 2)

        if len(coords) < 2:
            # Can't split, keep as is
            new_slots.append(slot)
            continue

        # Create sub-markers
        sub_markers = np.zeros(mask.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords[:num_pieces + 1], start=1):
            cv2.circle(sub_markers, (x, y), 2, i, -1)

        # Sub-watershed
        gradient = cv2.bitwise_not(cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        sub_labels = cv2.watershed(bgr, sub_markers)

        # Extract sub-slots
        for sub_id in range(1, len(coords[:num_pieces + 1]) + 1):
            sub_mask = ((sub_labels == sub_id) & (mask > 0)).astype(np.uint8) * 255
            sub_area = cv2.countNonZero(sub_mask)
            if sub_area < expected_area * 0.2:
                continue

            contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            ys, xs = np.where(sub_mask > 0)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(xs.mean())
                cy = int(ys.mean())

            new_slots.append({
                "label_id": slot["label_id"] * 10000 + sub_id,
                "area": sub_area,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "center": [cx, cy],
                "solidity": 0.85,
                "touches_border": slot["touches_border"],
                "contour": contour,
                "mask": sub_mask,
            })

    return new_slots


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_random_color_fill(shape: tuple, slots: list[dict]) -> np.ndarray:
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    np.random.seed(42)
    for i, slot in enumerate(slots):
        hue = int((i * 137) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 180, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
        result[slot["mask"] > 0] = color
    return result


def draw_overlay(base: np.ndarray, slots: list[dict]) -> np.ndarray:
    if len(base.shape) == 2:
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base.copy()

    for i, slot in enumerate(slots):
        hue = int((i * 137) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_tuple = tuple(int(c) for c in color)

        colored = np.zeros_like(overlay)
        colored[slot["mask"] > 0] = color_tuple
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.3, 0)
        cv2.drawContours(overlay, [slot["contour"]], -1, color_tuple, 1)

        cx, cy = slot["center"]
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

    return overlay


def draw_problem_regions(
    base: np.ndarray,
    merged: list[dict],
    fragments: list[dict],
) -> np.ndarray:
    """Highlight problem regions: merged in red, fragments in yellow."""
    if len(base.shape) == 2:
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base.copy()

    for slot in merged:
        colored = np.zeros_like(overlay)
        colored[slot["mask"] > 0] = (0, 0, 255)  # Red
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)
        cv2.drawContours(overlay, [slot["contour"]], -1, (0, 0, 255), 2)

    for slot in fragments:
        colored = np.zeros_like(overlay)
        colored[slot["mask"] > 0] = (0, 255, 255)  # Yellow
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)
        cv2.drawContours(overlay, [slot["contour"]], -1, (0, 255, 255), 2)

    return overlay


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment_puzzle(
    image_path: str,
    rows: int = 25,
    cols: int = 40,
    border_thickness: int = 10,
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    t0 = time.time()
    expected_count = rows * cols

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    height, width = img.shape
    expected_area = (height * width) / expected_count  # rough estimate
    print(f"Image: {width}x{height}")
    print(f"Grid: {cols}x{rows} = {expected_count} pieces")
    print(f"Expected piece area: ~{expected_area:.0f} px")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if output_dir is None:
        output_dir = str(Path(image_path).parent / "outputs" / "grid_watershed_v2")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    debug_dir = out if debug else None

    # ===================== STEP 1: Gap closing =====================
    print("\n--- Step 1: Aggressive gap closing ---")
    closed = aggressive_gap_close(binary, debug_dir=debug_dir)

    # ===================== STEP 2: Create wall/free masks =====================
    print("\n--- Step 2: Creating wall mask ---")

    # Add solid border around the image
    wall_mask = closed.copy()
    cv2.rectangle(wall_mask, (0, 0), (width - 1, height - 1), 255, border_thickness)

    free_space = cv2.bitwise_not(wall_mask)

    # Clean up: remove tiny free-space blobs that are noise
    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=8)
    min_component = int(expected_area * 0.15)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_component:
            free_space[labels_cc == i] = 0

    if debug:
        cv2.imwrite(str(out / "02_wall_mask.png"), wall_mask)
        cv2.imwrite(str(out / "03_free_space.png"), free_space)

    # ===================== STEP 3: Grid seeds =====================
    print("\n--- Step 3: Generating grid seeds ---")
    seeds = generate_grid_seeds(free_space, rows, cols, binary_mask=binary)
    print(f"  Seeds placed: {len(seeds)} / {expected_count}")

    if debug:
        seed_img = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        for sx, sy in seeds:
            cv2.circle(seed_img, (sx, sy), 5, (0, 0, 255), -1)
        cv2.imwrite(str(out / "04_seeds.png"), seed_img)

    # ===================== STEP 4: Watershed =====================
    print("\n--- Step 4: Watershed segmentation ---")
    labels = run_watershed(free_space, seeds, wall_mask)

    if debug:
        # Colorize labels for debug
        label_viz = np.zeros((height, width, 3), dtype=np.uint8)
        for lid in range(1, len(seeds) + 1):
            hue = int((lid * 137) % 180)
            color = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
            label_viz[labels == lid] = color
        cv2.imwrite(str(out / "05_watershed_raw.png"), label_viz)

    # ===================== STEP 5: Extract & validate =====================
    print("\n--- Step 5: Extracting and validating slots ---")
    good_slots, problem_slots = extract_and_validate_slots(
        labels, free_space, len(seeds), expected_area
    )

    merged = [s for s in problem_slots if s["area"] > expected_area * 2]
    fragments = [s for s in problem_slots if s["area"] < expected_area * 0.25]

    print(f"  Good slots: {len(good_slots)}")
    print(f"  Merged (too large): {len(merged)}")
    print(f"  Fragments (too small): {len(fragments)}")

    if debug and (merged or fragments):
        prob_img = draw_problem_regions(binary, merged, fragments)
        cv2.imwrite(str(out / "06_problems.png"), prob_img)

    # ===================== STEP 6: Re-split merged =====================
    if merged:
        print(f"\n--- Step 6: Re-splitting {len(merged)} merged regions ---")
        resplit = resplit_merged_regions(merged, free_space, expected_area, wall_mask)
        print(f"  Recovered {len(resplit)} slots from merged regions")
        good_slots.extend(resplit)

    # ===================== STEP 7: Final output =====================
    print(f"\n--- Step 7: Final output ---")
    all_slots = good_slots
    print(f"  Total slots: {len(all_slots)}")

    # Compute stats
    areas = [s["area"] for s in all_slots]
    median_area = float(np.median(areas)) if areas else 0

    # Still-merged check
    final_merged = [s for s in all_slots if s["area"] > median_area * 1.8]
    final_fragments = [s for s in all_slots if s["area"] < median_area * 0.3]

    # Save outputs
    color_fill = draw_random_color_fill(img.shape, all_slots)
    cv2.imwrite(str(out / "result_color_fill.png"), color_fill)

    overlay = draw_overlay(binary, all_slots)
    cv2.imwrite(str(out / "result_overlay.png"), overlay)

    if final_merged or final_fragments:
        prob_final = draw_problem_regions(binary, final_merged, final_fragments)
        cv2.imwrite(str(out / "result_problems.png"), prob_final)

    result = {
        "image": str(image_path),
        "grid": f"{cols}x{rows}",
        "expected_pieces": expected_count,
        "detected_pieces": len(all_slots),
        "seeds_placed": len(seeds),
        "area_stats": {
            "median": round(float(np.median(areas)), 1) if areas else 0,
            "mean": round(float(np.mean(areas)), 1) if areas else 0,
            "std": round(float(np.std(areas)), 1) if areas else 0,
            "min": int(min(areas)) if areas else 0,
            "max": int(max(areas)) if areas else 0,
        },
        "validation": {
            "good": len(good_slots),
            "still_merged": len(final_merged),
            "still_fragments": len(final_fragments),
            "merged_labels": [s["label_id"] for s in final_merged],
            "fragment_labels": [s["label_id"] for s in final_fragments],
        },
        "time_seconds": round(time.time() - t0, 2),
        "output_dir": str(out),
    }

    with open(str(out / "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone in {result['time_seconds']}s")
    print(f"Detected: {len(all_slots)} / {expected_count} pieces")
    if final_merged:
        print(f"WARNING: {len(final_merged)} still merged")
    if final_fragments:
        print(f"WARNING: {len(final_fragments)} still fragmented")
    print(f"Output: {out}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Grid-Prior Watershed v2")
    parser.add_argument("--image", required=True)
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--cols", type=int, default=40)
    parser.add_argument("--border-thickness", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = segment_puzzle(
        image_path=args.image,
        rows=args.rows,
        cols=args.cols,
        border_thickness=args.border_thickness,
        output_dir=args.output_dir,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
