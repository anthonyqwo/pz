import cv2
import numpy as np

def clean_and_count():
    line_mask_path = "d:/pz/data/boards/board_input_test/debug_import/line_mask.png"
    line_mask = cv2.imread(line_mask_path, cv2.IMREAD_GRAYSCALE)
    if line_mask is None:
        print("Error: Could not load line_mask.png")
        return
        
    print("--- Cleaning Experiments ---")
    
    # Let's try different methods
    # Method 1: Raw
    # Method 2: Area filtering of white components (remove very small noise lines/dots)
    # Method 3: Morphological opening (remove small white noise) followed by closing
    # Method 4: Combining connected components filtering
    
    # 1. Connected components filtering of line_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    print(f"Raw line_mask has {num_labels - 1} white connected components.")
    
    # Let's see area distribution of white components
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    areas.sort(reverse=True)
    print(f"Top 10 component areas: {areas[:10]}")
    print(f"Number of components with area < 10: {sum(1 for a in areas if a < 10)}")
    print(f"Number of components with area < 50: {sum(1 for a in areas if a < 50)}")
    print(f"Number of components with area < 100: {sum(1 for a in areas if a < 100)}")
    
    # Remove components with area < 50
    filtered_mask = np.zeros_like(line_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            filtered_mask[labels == i] = 255
            
    # Now let's run distance transform on cv2.bitwise_not(filtered_mask)
    free_space = cv2.bitwise_not(filtered_mask)
    cv2.rectangle(free_space, (0, 0), (free_space.shape[1]-1, free_space.shape[0]-1), 0, 8)
    
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    print(f"\nAfter removing small white components (area < 50):")
    print(f"Max distance: {max_dist}")
    
    for pct in [0.3, 0.4, 0.5, 0.6]:
        min_dist = max(15, int(max_dist * pct))
        kernel_size = min_dist if min_dist % 2 == 1 else min_dist + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
        local_max = cv2.dilate(dist_smooth, kernel)
        peaks_mask = (dist_smooth == local_max) & (dist_smooth > 5)
        num_peaks, _, _, _ = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
        print(f"  Factor {pct}: {num_peaks - 1} peaks (seeds)")

if __name__ == "__main__":
    clean_and_count()
