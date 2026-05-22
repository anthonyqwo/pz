import cv2
import numpy as np

def analyze():
    line_mask_path = "d:/pz/data/boards/board_input_test/debug_import/line_mask.png"
    rectified_path = "d:/pz/data/boards/board_input_test/debug_import/rectified.png"
    
    line_mask = cv2.imread(line_mask_path, cv2.IMREAD_GRAYSCALE)
    rectified = cv2.imread(rectified_path)
    
    if line_mask is None or rectified is None:
        print("Error: Could not load images.")
        return
        
    print(f"line_mask shape: {line_mask.shape}")
    print(f"rectified shape: {rectified.shape}")
    
    # 1. Check percentage of white pixels in line_mask
    white_pixels = np.count_nonzero(line_mask)
    total_pixels = line_mask.size
    print(f"White pixels: {white_pixels} ({white_pixels/total_pixels*100:.2f}%)")
    
    # 2. Run distance transform on cv2.bitwise_not(line_mask)
    free_space = cv2.bitwise_not(line_mask)
    # Add a border wall of thickness 8
    border_thickness = 8
    cv2.rectangle(free_space, (0, 0), (free_space.shape[1]-1, free_space.shape[0]-1), 0, border_thickness)
    
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    print(f"Max distance transform value: {max_dist}")
    
    # Let's count peak detections with different thresholds/parameters
    for pct in [0.3, 0.4, 0.5, 0.6]:
        min_dist = max(15, int(max_dist * pct))
        kernel_size = min_dist if min_dist % 2 == 1 else min_dist + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
        local_max = cv2.dilate(dist_smooth, kernel)
        peaks_mask = (dist_smooth == local_max) & (dist_smooth > 5)
        num_peaks, _, _, _ = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
        print(f"Threshold factor {pct}: {num_peaks - 1} peaks detected")

if __name__ == "__main__":
    analyze()
