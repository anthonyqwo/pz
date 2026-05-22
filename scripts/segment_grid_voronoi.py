"""
Grid-Prior Voronoi Puzzle Segmentation v3
==========================================
Completely different approach from v1/v2:

1. GENTLE gap closing on original binary (small kernel only)
2. Detect grid alignment via projection profiles
3. For each grid cell, DRAW the ideal grid lines where walls SHOULD be
4. Combine original walls + grid prior walls with weighting
5. Use connected components with grid-cell assignment
6. Post-process: force merge fragments, split merges

Key insight: We KNOW the grid layout (40x25). Instead of relying solely on
the noisy walls, we overlay a perfect grid and use it to break through gaps.
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
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Step 1: Gentle gap closing
# ---------------------------------------------------------------------------

def gentle_gap_close(mask: np.ndarray) -> np.ndarray:
    """Light morphological closing - don't over-thicken."""
    result = mask.copy()
    
    # Just a small square close to bridge 1-2 pixel gaps
    k3 = np.ones((3, 3), dtype=np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k3, iterations=1)
    
    # Short directional closes for slightly larger gaps  
    for angle in [0, 90]:
        kernel = _line_kernel(7, angle)
        closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        result = cv2.bitwise_or(result, closed)
    
    return result


def _line_kernel(length: int, angle_deg: int) -> np.ndarray:
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
# Step 2: Detect grid alignment via projection profiles  
# ---------------------------------------------------------------------------

def detect_grid_lines(
    binary: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find the actual grid line positions by analyzing projection profiles.
    
    Returns:
        h_lines: x-coordinates of vertical grid lines (cols+1)
        v_lines: y-coordinates of horizontal grid lines (rows+1)
        cell_w: average cell width
        cell_h: average cell height
    """
    height, width = binary.shape
    
    # Horizontal projection (sum each column -> detects vertical lines)
    h_proj = np.sum(binary > 0, axis=0).astype(float)
    # Vertical projection (sum each row -> detects horizontal lines)
    v_proj = np.sum(binary > 0, axis=1).astype(float)
    
    # Smooth
    from scipy.ndimage import uniform_filter1d
    h_smooth = uniform_filter1d(h_proj, size=7)
    v_smooth = uniform_filter1d(v_proj, size=7)
    
    # Expected spacings
    expected_cw = width / cols
    expected_ch = height / rows
    
    # Find vertical grid lines (peaks in horizontal projection)
    h_min_dist = int(expected_cw * 0.6)
    h_threshold = np.percentile(h_smooth, 50)
    h_peaks, _ = find_peaks(h_smooth, distance=h_min_dist, height=h_threshold)
    
    # Find horizontal grid lines (peaks in vertical projection)
    v_min_dist = int(expected_ch * 0.6)
    v_threshold = np.percentile(v_smooth, 50)
    v_peaks, _ = find_peaks(v_smooth, distance=v_min_dist, height=v_threshold)
    
    # Refine: we expect cols+1 vertical lines and rows+1 horizontal lines
    # If we found close to right number, use them; otherwise fall back to uniform
    
    if len(h_peaks) >= cols * 0.7:
        # Good detection - interpolate missing lines
        h_lines = _refine_grid_lines(h_peaks, cols + 1, width)
    else:
        h_lines = np.linspace(0, width - 1, cols + 1)
    
    if len(v_peaks) >= rows * 0.7:
        v_lines = _refine_grid_lines(v_peaks, rows + 1, height)
    else:
        v_lines = np.linspace(0, height - 1, rows + 1)
    
    cell_w = np.median(np.diff(h_lines))
    cell_h = np.median(np.diff(v_lines))
    
    return h_lines, v_lines, float(cell_w), float(cell_h)


def _refine_grid_lines(peaks: np.ndarray, expected_count: int, total_length: int) -> np.ndarray:
    """
    Given detected peaks, create a regular grid that best fits them.
    Use linear regression on the peaks to find the best offset + spacing.
    """
    peaks = np.sort(peaks).astype(float)
    
    if len(peaks) < 3:
        return np.linspace(0, total_length - 1, expected_count)
    
    # Estimate spacing from median of differences
    diffs = np.diff(peaks)
    spacing = np.median(diffs)
    
    # Assign each peak to the nearest grid index
    # Grid: line_i = offset + i * spacing
    # For each peak p, best_i = round((p - offset) / spacing)
    # We need to find the best offset
    
    # Try: offset = peaks[0] mod spacing
    offset = peaks[0] % spacing
    
    # Assign peaks to indices
    indices = np.round((peaks - offset) / spacing).astype(int)
    
    # Remove duplicate indices
    unique_mask = np.concatenate([[True], np.diff(indices) > 0])
    indices = indices[unique_mask]
    peaks_clean = peaks[unique_mask]
    
    if len(peaks_clean) >= 3:
        # Least-squares fit: peak = offset + index * spacing
        A = np.vstack([indices, np.ones(len(indices))]).T
        result = np.linalg.lstsq(A, peaks_clean, rcond=None)
        spacing_fit, offset_fit = result[0]
    else:
        spacing_fit = spacing
        offset_fit = offset
    
    # Generate full grid
    lines = offset_fit + np.arange(expected_count) * spacing_fit
    
    # Ensure lines are within bounds
    lines = np.clip(lines, 0, total_length - 1)
    
    # Make sure first and last are at edges
    lines[0] = 0
    lines[-1] = total_length - 1
    
    return lines


# ---------------------------------------------------------------------------
# Step 3: Build grid wall prior
# ---------------------------------------------------------------------------

def build_grid_prior_walls(
    shape: tuple[int, int],
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw ideal grid lines where walls SHOULD be.
    This creates a perfect grid that we'll combine with the actual walls.
    """
    grid = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    
    # Draw vertical lines
    for x in h_lines:
        x = int(round(x))
        if 0 <= x < width:
            cv2.line(grid, (x, 0), (x, height - 1), 255, thickness)
    
    # Draw horizontal lines
    for y in v_lines:
        y = int(round(y))
        if 0 <= y < height:
            cv2.line(grid, (0, y), (width - 1, y), 255, thickness)
    
    return grid


# ---------------------------------------------------------------------------
# Step 4: Combine walls with grid prior
# ---------------------------------------------------------------------------

def combine_walls(
    original_walls: np.ndarray,
    grid_prior: np.ndarray,
    agreement_radius: int = 15,
) -> np.ndarray:
    """
    Smart combination of original walls and grid prior:
    - Where original walls exist near grid lines -> definitely a wall (strengthen)
    - Where original walls exist far from grid lines -> likely a puzzle tab edge (keep)
    - Where grid lines exist but no original wall nearby -> possible gap (add thin wall)
    """
    height, width = original_walls.shape
    
    # Dilate original walls to create "agreement zone"
    dk = np.ones((agreement_radius, agreement_radius), dtype=np.uint8)
    orig_dilated = cv2.dilate(original_walls, dk, iterations=1)
    
    # Case 1: Grid prior where original walls agree (within radius)
    grid_confirmed = cv2.bitwise_and(grid_prior, orig_dilated)
    
    # Case 2: Grid prior where NO original wall nearby - these are gap fills
    # Make them thinner since we're less sure
    grid_gap_fill = cv2.bitwise_and(grid_prior, cv2.bitwise_not(orig_dilated))
    # Thin the gap-fill lines
    thin_kernel = np.ones((3, 3), dtype=np.uint8)
    grid_gap_fill = cv2.erode(grid_gap_fill, thin_kernel, iterations=1)
    
    # Combine: original walls + confirmed grid + gap fills
    combined = cv2.bitwise_or(original_walls, grid_confirmed)
    combined = cv2.bitwise_or(combined, grid_gap_fill)
    
    return combined


# ---------------------------------------------------------------------------
# Step 5: Label by grid cell assignment
# ---------------------------------------------------------------------------

def label_by_grid_cells(
    free_space: np.ndarray,
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Assign each free pixel to its grid cell based on position.
    This is a Voronoi-like assignment using the grid centers.
    
    Much faster and more robust than watershed when we know the grid.
    """
    height, width = free_space.shape
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Create grid cell centers
    centers = []
    for r in range(rows):
        y_center = (v_lines[r] + v_lines[r + 1]) / 2
        for c in range(cols):
            x_center = (h_lines[c] + h_lines[c + 1]) / 2
            centers.append((x_center, y_center))
    
    # For each pixel, find which grid cell it belongs to
    # Use vectorized approach: compute cell indices directly
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Find column index for each x
    col_indices = np.searchsorted(h_lines, x_coords, side='right') - 1
    col_indices = np.clip(col_indices, 0, cols - 1)
    
    # Find row index for each y
    row_indices = np.searchsorted(v_lines, y_coords, side='right') - 1
    row_indices = np.clip(row_indices, 0, rows - 1)
    
    # Create label map: label = row * cols + col + 1 (1-indexed)
    row_map = row_indices[:, np.newaxis]  # (height, 1)
    col_map = col_indices[np.newaxis, :]  # (1, width)
    label_map = row_map * cols + col_map + 1  # (height, width)
    
    # Apply only to free space
    labels = np.where(free_space > 0, label_map.astype(np.int32), 0)
    
    return labels


# ---------------------------------------------------------------------------
# Step 6: Refine labels using connected components
# ---------------------------------------------------------------------------

def refine_with_connected_components(
    labels: np.ndarray,
    free_space: np.ndarray,
    rows: int,
    cols: int,
    expected_area: float,
) -> np.ndarray:
    """
    The grid-cell assignment may create disconnected regions within
    the same label (due to wall topology). Fix this by:
    1. For each label, check connected components
    2. Keep the largest component with that label
    3. Reassign small disconnected pieces to neighboring labels
    """
    height, width = labels.shape
    expected_count = rows * cols
    
    refined = labels.copy()
    
    for label_id in range(1, expected_count + 1):
        mask = (refined == label_id).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            continue
        
        num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_cc <= 2:  # Background + 1 component = OK
            continue
        
        # Find the largest component (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_cc = np.argmax(areas) + 1
        
        # Reassign smaller components to 0 (will be picked up by neighbor expansion)
        for cc_id in range(1, num_cc):
            if cc_id != largest_cc:
                refined[cc_labels == cc_id] = 0
    
    # Now fill in the gaps (pixels that were set to 0)
    # Use iterative dilation of existing labels to fill
    iterations = 0
    max_iter = 50
    while True:
        unlabeled = (refined == 0) & (free_space > 0)
        unlabeled_count = np.count_nonzero(unlabeled)
        if unlabeled_count == 0 or iterations >= max_iter:
            break
        
        # Dilate each label by 1 pixel
        for label_id in range(1, expected_count + 1):
            mask = (refined == label_id).astype(np.uint8)
            dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            # Only fill unlabeled free-space pixels
            fill_mask = (dilated > 0) & unlabeled
            refined[fill_mask] = label_id
            # Update unlabeled
            unlabeled = (refined == 0) & (free_space > 0)
        
        iterations += 1
    
    return refined


def refine_with_connected_components_fast(
    labels: np.ndarray,
    free_space: np.ndarray,
    rows: int,
    cols: int,
    expected_area: float,
) -> np.ndarray:
    """
    Faster version: instead of iterating per-label for gap filling,
    use a single nearest-label expansion.
    """
    height, width = labels.shape
    expected_count = rows * cols
    
    refined = labels.copy()
    
    # Step 1: For each label, keep only largest connected component
    for label_id in range(1, expected_count + 1):
        mask = (refined == label_id).astype(np.uint8)
        area = cv2.countNonZero(mask)
        if area == 0:
            continue
        
        num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_cc <= 2:  # Background + 1 component = OK
            continue
        
        # Keep only the largest connected component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_cc = np.argmax(areas) + 1
        
        for cc_id in range(1, num_cc):
            if cc_id != largest_cc:
                refined[cc_labels == cc_id] = 0
    
    # Step 2: Fill gaps using distance-based nearest neighbor
    unlabeled = (refined == 0) & (free_space > 0)
    unlabeled_count = np.count_nonzero(unlabeled)
    print(f"  Pixels to reassign: {unlabeled_count}")
    
    if unlabeled_count > 0:
        # Use iterative morphological expansion (much faster per iteration)
        kernel = np.ones((3, 3), dtype=np.uint8)
        for iteration in range(100):
            unlabeled = (refined == 0) & (free_space > 0)
            if np.count_nonzero(unlabeled) == 0:
                break
            
            # For each unlabeled pixel, find the label of any labeled neighbor
            # This is equivalent to 1-pixel dilation of all labeled regions simultaneously
            
            # Shift in 8 directions and take the first non-zero label found
            padded = np.pad(refined, 1, mode='constant', constant_values=0)
            neighbors = np.zeros((height, width, 8), dtype=np.int32)
            
            # 8-connectivity neighbors
            neighbors[:, :, 0] = padded[0:height, 0:width]      # top-left
            neighbors[:, :, 1] = padded[0:height, 1:width+1]    # top
            neighbors[:, :, 2] = padded[0:height, 2:width+2]    # top-right
            neighbors[:, :, 3] = padded[1:height+1, 0:width]    # left
            neighbors[:, :, 4] = padded[1:height+1, 2:width+2]  # right
            neighbors[:, :, 5] = padded[2:height+2, 0:width]    # bottom-left
            neighbors[:, :, 6] = padded[2:height+2, 1:width+1]  # bottom
            neighbors[:, :, 7] = padded[2:height+2, 2:width+2]  # bottom-right
            
            # For unlabeled pixels, take the most common non-zero neighbor
            for n in range(8):
                can_fill = unlabeled & (neighbors[:, :, n] > 0)
                refined[can_fill] = neighbors[:, :, n][can_fill]
                unlabeled = (refined == 0) & (free_space > 0)
        
        remaining = np.count_nonzero((refined == 0) & (free_space > 0))
        print(f"  Remaining unlabeled: {remaining}")
    
    return refined


# ---------------------------------------------------------------------------
# Step 7: Visualization
# ---------------------------------------------------------------------------

def draw_random_color_fill(shape, labels, max_label):
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    np.random.seed(42)
    for label_id in range(1, max_label + 1):
        mask = labels == label_id
        if not np.any(mask):
            continue
        hue = int((label_id * 137) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 180, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
        result[mask] = color
    return result


def draw_grid_overlay(base, h_lines, v_lines):
    if len(base.shape) == 2:
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base.copy()
    
    height, width = overlay.shape[:2]
    
    for x in h_lines:
        x = int(round(x))
        cv2.line(overlay, (x, 0), (x, height - 1), (0, 0, 255), 1)
    
    for y in v_lines:
        y = int(round(y))
        cv2.line(overlay, (0, y), (width - 1, y), (0, 0, 255), 1)
    
    return overlay


def compute_slot_stats(labels, free_space, rows, cols):
    """Compute per-slot statistics for validation."""
    expected_count = rows * cols
    slots = []
    
    for label_id in range(1, expected_count + 1):
        mask = ((labels == label_id) & (free_space > 0)).astype(np.uint8)
        area = cv2.countNonZero(mask)
        if area == 0:
            slots.append({"label_id": label_id, "area": 0, "status": "missing"})
            continue
        
        ys, xs = np.where(mask > 0)
        bbox = [int(xs.min()), int(ys.min()), 
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        
        # Check connectivity
        num_cc, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        is_connected = num_cc <= 2
        
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        
        solidity = 0
        if contour is not None:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / hull_area if hull_area > 0 else 0
        
        slots.append({
            "label_id": label_id,
            "area": area,
            "bbox": bbox,
            "connected": is_connected,
            "solidity": round(solidity, 3),
            "status": "ok",
        })
    
    return slots


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment_puzzle(
    image_path: str,
    rows: int = 25,
    cols: int = 40,
    border_thickness: int = 8,
    grid_wall_thickness: int = 3,
    agreement_radius: int = 15,
    output_dir: str | None = None,
    debug: bool = False,
) -> dict:
    t0 = time.time()
    expected_count = rows * cols
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    
    height, width = img.shape
    expected_area = (height * width) / expected_count
    print(f"Image: {width}x{height}")
    print(f"Grid: {cols}x{rows} = {expected_count} pieces")
    print(f"Expected piece area: ~{expected_area:.0f} px")
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    if output_dir is None:
        output_dir = str(Path(image_path).parent / "outputs" / "grid_voronoi")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ===================== STEP 1: Gentle gap closing =====================
    print("\n--- Step 1: Gentle gap closing ---")
    closed = gentle_gap_close(binary)
    
    if debug:
        cv2.imwrite(str(out / "01_closed.png"), closed)
    
    # ===================== STEP 2: Detect grid alignment =====================
    print("\n--- Step 2: Detecting grid alignment ---")
    h_lines, v_lines, cell_w, cell_h = detect_grid_lines(closed, rows, cols)
    print(f"  Cell size: {cell_w:.1f} x {cell_h:.1f}")
    print(f"  Vertical lines detected: {len(h_lines)}")
    print(f"  Horizontal lines detected: {len(v_lines)}")
    
    if debug:
        grid_overlay = draw_grid_overlay(binary, h_lines, v_lines)
        cv2.imwrite(str(out / "02_grid_overlay.png"), grid_overlay)
    
    # ===================== STEP 3: Build grid prior walls =====================
    print("\n--- Step 3: Building grid prior walls ---")
    grid_prior = build_grid_prior_walls(
        (height, width), h_lines, v_lines, thickness=grid_wall_thickness
    )
    
    if debug:
        cv2.imwrite(str(out / "03_grid_prior.png"), grid_prior)
    
    # ===================== STEP 4: Combine walls =====================
    print("\n--- Step 4: Combining walls ---")
    combined = combine_walls(closed, grid_prior, agreement_radius=agreement_radius)
    
    # Add solid border
    cv2.rectangle(combined, (0, 0), (width - 1, height - 1), 255, border_thickness)
    
    if debug:
        cv2.imwrite(str(out / "04_combined_walls.png"), combined)
    
    # Free space
    free_space = cv2.bitwise_not(combined)
    
    if debug:
        cv2.imwrite(str(out / "05_free_space.png"), free_space)
    
    # ===================== STEP 5: Label by grid cells =====================
    print("\n--- Step 5: Labeling by grid cells ---")
    labels = label_by_grid_cells(free_space, h_lines, v_lines, rows, cols)
    
    unique_labels = len(np.unique(labels[labels > 0]))
    print(f"  Initial labels: {unique_labels}")
    
    if debug:
        color_raw = draw_random_color_fill(img.shape, labels, expected_count)
        cv2.imwrite(str(out / "06_labels_raw.png"), color_raw)
    
    # ===================== STEP 6: Refine with CC =====================
    print("\n--- Step 6: Refining with connected components ---")
    labels = refine_with_connected_components_fast(
        labels, free_space, rows, cols, expected_area
    )
    
    unique_labels = len(np.unique(labels[labels > 0]))
    print(f"  Refined labels: {unique_labels}")
    
    # ===================== STEP 7: Output =====================
    print("\n--- Step 7: Saving results ---")
    
    # Color fill
    color_fill = draw_random_color_fill(img.shape, labels, expected_count)
    cv2.imwrite(str(out / "result_color_fill.png"), color_fill)
    
    # Compute statistics
    slot_stats = compute_slot_stats(labels, free_space, rows, cols)
    
    ok_slots = [s for s in slot_stats if s["status"] == "ok" and s["area"] > 0]
    missing_slots = [s for s in slot_stats if s["area"] == 0]
    
    areas = [s["area"] for s in ok_slots]
    median_area = float(np.median(areas)) if areas else 0
    
    merged = [s for s in ok_slots if s["area"] > median_area * 1.8]
    fragments = [s for s in ok_slots if s["area"] < median_area * 0.3]
    disconnected = [s for s in ok_slots if not s.get("connected", True)]
    
    result = {
        "image": str(image_path),
        "grid": f"{cols}x{rows}",
        "expected_pieces": expected_count,
        "detected_pieces": len(ok_slots),
        "missing_pieces": len(missing_slots),
        "cell_size": {"w": round(cell_w, 1), "h": round(cell_h, 1)},
        "area_stats": {
            "median": round(median_area, 1),
            "mean": round(float(np.mean(areas)), 1) if areas else 0,
            "std": round(float(np.std(areas)), 1) if areas else 0,
            "min": int(min(areas)) if areas else 0,
            "max": int(max(areas)) if areas else 0,
        },
        "validation": {
            "ok": len(ok_slots),
            "missing": len(missing_slots),
            "merged_count": len(merged),
            "fragment_count": len(fragments),
            "disconnected_count": len(disconnected),
            "merged_labels": [s["label_id"] for s in merged],
            "fragment_labels": [s["label_id"] for s in fragments],
            "missing_labels": [s["label_id"] for s in missing_slots],
        },
        "time_seconds": round(time.time() - t0, 2),
        "output_dir": str(out),
    }
    
    with open(str(out / "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nDone in {result['time_seconds']}s")
    print(f"Detected: {len(ok_slots)} / {expected_count} pieces")
    print(f"Missing: {len(missing_slots)}")
    if merged:
        print(f"WARNING: {len(merged)} merged")
    if fragments:
        print(f"WARNING: {len(fragments)} fragments")
    if disconnected:
        print(f"WARNING: {len(disconnected)} disconnected")
    print(f"Output: {out}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Grid-Prior Voronoi v3")
    parser.add_argument("--image", required=True)
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--cols", type=int, default=40)
    parser.add_argument("--border-thickness", type=int, default=8)
    parser.add_argument("--grid-wall-thickness", type=int, default=3)
    parser.add_argument("--agreement-radius", type=int, default=15)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    result = segment_puzzle(
        image_path=args.image,
        rows=args.rows,
        cols=args.cols,
        border_thickness=args.border_thickness,
        grid_wall_thickness=args.grid_wall_thickness,
        agreement_radius=args.agreement_radius,
        output_dir=args.output_dir,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
