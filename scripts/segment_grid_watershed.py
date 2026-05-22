"""
Grid-Prior Watershed Puzzle Segmentation
=========================================
Segments a binary puzzle grid image (white lines on black) into individual
puzzle slots using a constrained watershed approach.

Known constraints:
  - Grid is 40 columns × 25 rows = 1000 pieces
  - Each piece has 4 sides with concave/convex tabs
  - No merged pieces allowed
  - No irregular shapes allowed
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
# Step 1: Morphological gap closing
# ---------------------------------------------------------------------------

def directional_close(mask: np.ndarray, length: int = 15, iterations: int = 1) -> np.ndarray:
    """
    Close gaps using directional (linear) structuring elements at 0°, 45°, 90°, 135°.
    This is more effective than a square kernel for line-like structures.
    """
    result = mask.copy()

    angles = [0, 45, 90, 135]
    for angle in angles:
        # Create a line structuring element
        kernel = _line_kernel(length, angle)
        closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        result = cv2.bitwise_or(result, closed)

    return result


def _line_kernel(length: int, angle_deg: int) -> np.ndarray:
    """Create a line-shaped structuring element at a given angle."""
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


def close_small_gaps(mask: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    """Standard morphological close with a small square kernel."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ---------------------------------------------------------------------------
# Step 2: Skeleton endpoint bridging (multi-round)
# ---------------------------------------------------------------------------

def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin the white lines to 1px skeleton using OpenCV morphology."""
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        binary = mask > 0
        skeleton = sk_skeletonize(binary).astype(np.uint8) * 255
        return skeleton
    except ImportError:
        # Fallback: OpenCV-based skeletonization
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


def find_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Find endpoints of skeleton (pixels with exactly 1 neighbor)."""
    binary = (skeleton > 0).astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((binary == 1) & (neighbor_count == 1))
    return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))


def bridge_endpoints(
    skeleton: np.ndarray,
    gap_limit: int = 40,
    max_rounds: int = 5,
) -> np.ndarray:
    """
    Multi-round endpoint bridging. Each round:
    1. Find all endpoints
    2. For each endpoint, find closest other endpoint within gap_limit
    3. Connect them with a line
    4. Re-skeletonize? No, just repeat endpoint finding.
    """
    output = skeleton.copy()

    for round_num in range(max_rounds):
        endpoints = find_endpoints(output)
        if len(endpoints) < 2:
            break

        # Build a KD-tree-like approach (simple for small counts)
        connected = 0
        used = set()

        # Sort by x then y for deterministic pairing
        endpoints_sorted = sorted(endpoints)

        for i, p1 in enumerate(endpoints_sorted):
            if p1 in used:
                continue

            best_dist = float('inf')
            best_p2 = None

            for j, p2 in enumerate(endpoints_sorted):
                if i == j or p2 in used:
                    continue

                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dist = (dx * dx + dy * dy) ** 0.5

                if dist < best_dist and 0 < dist <= gap_limit:
                    best_dist = dist
                    best_p2 = p2

            if best_p2 is not None:
                cv2.line(output, p1, best_p2, 255, 1)
                used.add(p1)
                used.add(best_p2)
                connected += 1

        if connected == 0:
            break

    return output


# ---------------------------------------------------------------------------
# Step 3: Grid-Prior seed generation
# ---------------------------------------------------------------------------

def generate_grid_seeds(
    free_space: np.ndarray,
    rows: int,
    cols: int,
    search_radius: int = 0,
) -> list[tuple[int, int]]:
    """
    Generate seeds at grid cell centers. If a center falls on a wall pixel,
    search nearby for the nearest free-space pixel.
    """
    height, width = free_space.shape
    cell_w = width / cols
    cell_h = height / rows

    if search_radius <= 0:
        search_radius = int(min(cell_w, cell_h) * 0.35)

    seeds = []
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)

            # Clamp to image bounds
            cx = min(max(0, cx), width - 1)
            cy = min(max(0, cy), height - 1)

            if free_space[cy, cx] > 0:
                seeds.append((cx, cy))
            else:
                # Search for nearest free pixel
                found = _find_nearest_free(free_space, cx, cy, search_radius)
                if found is not None:
                    seeds.append(found)
                else:
                    # Expand search
                    found = _find_nearest_free(free_space, cx, cy, search_radius * 2)
                    if found is not None:
                        seeds.append(found)

    return seeds


def _find_nearest_free(mask: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int] | None:
    """Find the nearest non-zero pixel to (cx, cy) within a square radius."""
    h, w = mask.shape
    best_dist = float('inf')
    best = None

    y1 = max(0, cy - radius)
    y2 = min(h - 1, cy + radius)
    x1 = max(0, cx - radius)
    x2 = min(w - 1, cx + radius)

    # Extract sub-region for efficiency
    sub = mask[y1:y2+1, x1:x2+1]
    ys, xs = np.where(sub > 0)

    if len(xs) == 0:
        return None

    # Find closest
    dists = (xs - (cx - x1)) ** 2 + (ys - (cy - y1)) ** 2
    idx = np.argmin(dists)
    return (int(xs[idx]) + x1, int(ys[idx]) + y1)


# ---------------------------------------------------------------------------
# Step 4: Watershed segmentation
# ---------------------------------------------------------------------------

def run_watershed(
    free_space: np.ndarray,
    seeds: list[tuple[int, int]],
    edge_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run marker-controlled watershed.

    Args:
        free_space: Binary mask where 255 = free space, 0 = wall
        seeds: List of (x, y) seed points
        edge_map: Optional gradient/edge image for watershed energy

    Returns:
        labels: Integer array where each pixel has its label (1-indexed)
    """
    # Create markers
    markers = np.zeros(free_space.shape, dtype=np.int32)
    for label_id, (x, y) in enumerate(seeds, start=1):
        markers[y, x] = label_id

    if edge_map is not None:
        # Use edge map as the gradient for watershed
        if len(edge_map.shape) == 2:
            bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        else:
            bgr = edge_map
    else:
        # Create a gradient image from the free space
        gray = cv2.bitwise_not(free_space)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    labels = cv2.watershed(bgr, markers)
    return labels


# ---------------------------------------------------------------------------
# Step 5: Slot extraction and validation
# ---------------------------------------------------------------------------

def extract_slots(
    labels: np.ndarray,
    free_space: np.ndarray,
    num_seeds: int,
    min_area: int = 500,
    max_area: int | None = None,
    border_margin: int = 5,
) -> list[dict]:
    """
    Extract individual slot masks from watershed labels.
    Validate each slot for reasonable shape.
    """
    height, width = labels.shape
    slots = []

    for label_id in range(1, num_seeds + 1):
        # Only count pixels that are in free space
        region_mask = ((labels == label_id) & (free_space > 0)).astype(np.uint8) * 255
        area = cv2.countNonZero(region_mask)

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        # Find bounding box
        ys, xs = np.where(region_mask > 0)
        if len(xs) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1

        # Check if touches border
        touches = (x_min <= border_margin or y_min <= border_margin or
                   x_max >= width - border_margin or y_max >= height - border_margin)

        # Find contour
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Center
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

        slots.append({
            "label_id": label_id,
            "area": area,
            "perimeter": round(perimeter, 1),
            "bbox": [x_min, y_min, bbox_w, bbox_h],
            "center": [cx, cy],
            "solidity": round(solidity, 3),
            "touches_border": touches,
            "contour": contour,
            "mask": region_mask,
        })

    return slots


def validate_slots(
    slots: list[dict],
    expected_count: int,
    median_area: float | None = None,
) -> dict:
    """
    Validate extracted slots against expectations.
    Returns validation report.
    """
    if not slots:
        return {"valid": False, "reason": "No slots found", "count": 0}

    areas = [s["area"] for s in slots]
    if median_area is None:
        median_area = float(np.median(areas))

    # Check for merged pieces (area >> median)
    merged = [s for s in slots if s["area"] > median_area * 1.8]

    # Check for fragments (area << median)
    fragments = [s for s in slots if s["area"] < median_area * 0.3]

    # Check solidity (puzzle pieces should be fairly solid, ~0.7-0.95)
    irregular = [s for s in slots if s["solidity"] < 0.5 or s["solidity"] > 0.99]

    return {
        "valid": len(merged) == 0 and len(fragments) == 0,
        "count": len(slots),
        "expected": expected_count,
        "median_area": round(median_area, 1),
        "area_std": round(float(np.std(areas)), 1),
        "merged_count": len(merged),
        "fragment_count": len(fragments),
        "irregular_count": len(irregular),
        "merged_labels": [s["label_id"] for s in merged],
        "fragment_labels": [s["label_id"] for s in fragments],
    }


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------

def draw_overlay(
    base_image: np.ndarray,
    slots: list[dict],
    show_labels: bool = True,
    show_centers: bool = True,
) -> np.ndarray:
    """Draw colored overlay of all slots on the base image."""
    if len(base_image.shape) == 2:
        overlay = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base_image.copy()

    # Generate distinct colors
    np.random.seed(42)
    colors = []
    for i in range(len(slots)):
        hue = int((i * 137) % 180)  # Golden angle for color spread
        color = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))

    for i, slot in enumerate(slots):
        color = colors[i % len(colors)]

        # Draw filled contour with transparency
        mask = slot["mask"]
        colored = np.zeros_like(overlay)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.3, 0)

        # Draw contour outline
        cv2.drawContours(overlay, [slot["contour"]], -1, color, 1)

        if show_centers:
            cx, cy = slot["center"]
            cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

        if show_labels:
            cx, cy = slot["center"]
            label_text = str(slot["label_id"])
            cv2.putText(overlay, label_text, (cx - 10, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return overlay


def draw_random_color_fill(
    shape: tuple[int, int],
    slots: list[dict],
) -> np.ndarray:
    """Fill each slot with a random solid color for easy visual check."""
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

    np.random.seed(42)
    for i, slot in enumerate(slots):
        hue = int((i * 137) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 180, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
        result[slot["mask"] > 0] = color

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment_puzzle(
    image_path: str,
    rows: int = 25,
    cols: int = 40,
    gap_close_kernel: int = 7,
    gap_close_iter: int = 2,
    directional_length: int = 15,
    bridge_gap_limit: int = 40,
    bridge_rounds: int = 5,
    wall_dilate: int = 3,
    border_thickness: int = 8,
    min_slot_area: int = 500,
    max_slot_area: int | None = None,
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    """
    Main segmentation pipeline.

    Args:
        image_path: Path to binary puzzle image
        rows: Number of puzzle rows
        cols: Number of puzzle columns
        gap_close_kernel: Kernel size for initial morphological close
        gap_close_iter: Iterations for initial close
        directional_length: Length of directional closing kernels
        bridge_gap_limit: Maximum distance for endpoint bridging
        bridge_rounds: Number of bridging rounds
        wall_dilate: Dilation of final wall mask before watershed
        border_thickness: Thickness of image border wall
        min_slot_area: Minimum valid slot area
        max_slot_area: Maximum valid slot area
        output_dir: Directory for output files
        debug: Save debug images
    """
    t0 = time.time()
    expected_count = rows * cols

    # --- Load image ---
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    height, width = img.shape
    print(f"Image: {width}x{height}")
    print(f"Grid: {cols}x{rows} = {expected_count} pieces")

    # Clean binarize
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # --- Output setup ---
    if output_dir is None:
        output_dir = str(Path(image_path).parent / "outputs" / "grid_watershed")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Gap closing ---
    print("Step 1: Morphological gap closing...")

    # 1a. Standard close
    closed = close_small_gaps(binary, kernel_size=gap_close_kernel, iterations=gap_close_iter)

    # 1b. Directional close
    closed = directional_close(closed, length=directional_length, iterations=1)

    if debug:
        cv2.imwrite(str(out_path / "01_gap_closed.png"), closed)

    # --- Step 2: Skeleton + endpoint bridging ---
    print("Step 2: Skeletonize + endpoint bridging...")
    skeleton = skeletonize(closed)

    if debug:
        cv2.imwrite(str(out_path / "02_skeleton.png"), skeleton)
        endpoints = find_endpoints(skeleton)
        print(f"  Endpoints before bridging: {len(endpoints)}")

    bridged = bridge_endpoints(skeleton, gap_limit=bridge_gap_limit, max_rounds=bridge_rounds)

    if debug:
        endpoints_after = find_endpoints(bridged)
        print(f"  Endpoints after bridging: {len(endpoints_after)}")
        cv2.imwrite(str(out_path / "03_bridged.png"), bridged)

    # --- Step 3: Prepare wall mask ---
    print("Step 3: Preparing wall mask...")

    # Dilate skeleton back to create walls
    wall_mask = bridged.copy()
    if wall_dilate > 0:
        dk = np.ones((wall_dilate, wall_dilate), dtype=np.uint8)
        wall_mask = cv2.dilate(wall_mask, dk, iterations=1)

    # Add border
    cv2.rectangle(wall_mask, (0, 0), (width - 1, height - 1), 255, border_thickness)

    # Free space = NOT wall
    free_space = cv2.bitwise_not(wall_mask)

    if debug:
        cv2.imwrite(str(out_path / "04_wall_mask.png"), wall_mask)
        cv2.imwrite(str(out_path / "05_free_space.png"), free_space)

    # --- Step 4: Generate grid seeds ---
    print("Step 4: Generating grid seeds...")
    seeds = generate_grid_seeds(free_space, rows, cols)
    print(f"  Seeds placed: {len(seeds)} / {expected_count}")

    if debug:
        seed_img = cv2.cvtColor(free_space, cv2.COLOR_GRAY2BGR)
        for sx, sy in seeds:
            cv2.circle(seed_img, (sx, sy), 4, (0, 0, 255), -1)
        cv2.imwrite(str(out_path / "06_seeds.png"), seed_img)

    # --- Step 5: Watershed ---
    print("Step 5: Running watershed...")

    # Create edge energy: use distance transform of free space
    # The gradient is the negated distance (walls are high, centers are low)
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Invert so walls have high values (act as ridges for watershed)
    gradient = 255 - dist_norm
    gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    labels = run_watershed(free_space, seeds, gradient_bgr)

    if debug:
        cv2.imwrite(str(out_path / "07_distance.png"), dist_norm)

    # --- Step 6: Extract slots ---
    print("Step 6: Extracting slots...")
    slots = extract_slots(
        labels, free_space, len(seeds),
        min_area=min_slot_area,
        max_area=max_slot_area,
        border_margin=border_thickness,
    )
    print(f"  Valid slots: {len(slots)}")

    # --- Step 7: Validate ---
    print("Step 7: Validating...")
    validation = validate_slots(slots, expected_count)
    print(f"  Validation: {json.dumps({k: v for k, v in validation.items() if k != 'merged_labels' and k != 'fragment_labels'}, indent=2)}")

    if validation["merged_count"] > 0:
        print(f"  WARNING: {validation['merged_count']} merged regions detected!")
    if validation["fragment_count"] > 0:
        print(f"  WARNING: {validation['fragment_count']} fragments detected!")

    # --- Step 8: Output ---
    print("Step 8: Saving results...")

    # Random color fill for easy visual inspection
    color_fill = draw_random_color_fill(img.shape, slots)
    cv2.imwrite(str(out_path / "result_color_fill.png"), color_fill)

    # Overlay on original
    overlay = draw_overlay(binary, slots, show_labels=True, show_centers=True)
    cv2.imwrite(str(out_path / "result_overlay.png"), overlay)

    # Slot statistics
    areas = [s["area"] for s in slots]
    solidities = [s["solidity"] for s in slots]

    result = {
        "image": str(image_path),
        "grid": f"{cols}x{rows}",
        "expected_pieces": expected_count,
        "detected_pieces": len(slots),
        "seeds_placed": len(seeds),
        "area_stats": {
            "median": round(float(np.median(areas)), 1) if areas else 0,
            "mean": round(float(np.mean(areas)), 1) if areas else 0,
            "std": round(float(np.std(areas)), 1) if areas else 0,
            "min": int(min(areas)) if areas else 0,
            "max": int(max(areas)) if areas else 0,
        },
        "solidity_stats": {
            "median": round(float(np.median(solidities)), 3) if solidities else 0,
            "min": round(float(min(solidities)), 3) if solidities else 0,
            "max": round(float(max(solidities)), 3) if solidities else 0,
        },
        "validation": validation,
        "time_seconds": round(time.time() - t0, 2),
        "output_dir": str(out_path),
    }

    # Save JSON
    with open(str(out_path / "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone in {result['time_seconds']}s")
    print(f"Detected: {len(slots)} / {expected_count} pieces")
    print(f"Output: {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid-Prior Watershed Puzzle Segmentation")
    parser.add_argument("--image", required=True, help="Path to binary puzzle image")
    parser.add_argument("--rows", type=int, default=25, help="Number of rows")
    parser.add_argument("--cols", type=int, default=40, help="Number of columns")
    parser.add_argument("--gap-close-kernel", type=int, default=7)
    parser.add_argument("--gap-close-iter", type=int, default=2)
    parser.add_argument("--directional-length", type=int, default=15)
    parser.add_argument("--bridge-gap-limit", type=int, default=40)
    parser.add_argument("--bridge-rounds", type=int, default=5)
    parser.add_argument("--wall-dilate", type=int, default=3)
    parser.add_argument("--border-thickness", type=int, default=8)
    parser.add_argument("--min-slot-area", type=int, default=500)
    parser.add_argument("--max-slot-area", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = segment_puzzle(
        image_path=args.image,
        rows=args.rows,
        cols=args.cols,
        gap_close_kernel=args.gap_close_kernel,
        gap_close_iter=args.gap_close_iter,
        directional_length=args.directional_length,
        bridge_gap_limit=args.bridge_gap_limit,
        bridge_rounds=args.bridge_rounds,
        wall_dilate=args.wall_dilate,
        border_thickness=args.border_thickness,
        min_slot_area=args.min_slot_area,
        max_slot_area=args.max_slot_area,
        output_dir=args.output_dir,
        debug=args.debug,
    )

    print("\n" + json.dumps(
        {k: v for k, v in result.items() if k != "slots"},
        ensure_ascii=False,
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
