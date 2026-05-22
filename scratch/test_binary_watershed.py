import cv2
import numpy as np

def run_test():
    img = cv2.imread("d:/pz/output.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load output.jpg")
        return
        
    print("--- Binary Watershed Test ---")
    
    # 1. Clean the binary grid mask by keeping only components with area > 1000
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    grid_mask = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            grid_mask[labels == i] = 255
            
    print(f"Cleaned grid mask white pixels: {np.count_nonzero(grid_mask)}")
    
    # 2. Add an outer boundary wall (thickness 8)
    border_thickness = 8
    cv2.rectangle(grid_mask, (0, 0), (grid_mask.shape[1]-1, grid_mask.shape[0]-1), 255, border_thickness)
    
    # 3. Find black pieces (free space)
    free_space = cv2.bitwise_not(grid_mask)
    
    # 4. Run distance transform
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    print(f"Max distance in pieces: {max_dist}")
    
    # 5. Detect seeds (local maxima)
    # Using a robust window size
    min_dist = max(15, int(max_dist * 0.5))
    kernel_size = min_dist if min_dist % 2 == 1 else min_dist + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    local_max = cv2.dilate(dist_smooth, kernel)
    peaks_mask = (dist_smooth == local_max) & (dist_smooth > 5)
    
    num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
    print(f"Detected {num_peaks - 1} seeds (pieces).")
    
    # 6. Run watershed
    markers = np.zeros(free_space.shape, dtype=np.int32)
    for label_id in range(1, num_peaks):
        cx, cy = peak_centroids[label_id]
        px, py = int(round(cx)), int(round(cy))
        markers[py, px] = label_id
        
    bgr_for_watershed = cv2.cvtColor(grid_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr_for_watershed, markers)
    
    # 7. Extract slots
    slots_count = 0
    for label_id in range(1, num_peaks):
        mask = (markers == label_id) & (free_space > 0)
        area = np.count_nonzero(mask)
        if area > 1000:
            slots_count += 1
            
    print(f"Extracted {slots_count} slots (area > 1000).")

if __name__ == "__main__":
    run_test()
