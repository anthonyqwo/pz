import cv2
import numpy as np

def visualize():
    img = cv2.imread("d:/pz/output.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
        
    # Clean the binary grid mask by keeping only components with area > 1000
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    grid_mask = np.zeros_like(img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 1000:
            grid_mask[labels == i] = 255
            
    # Add an outer boundary wall (thickness 8)
    border_thickness = 8
    cv2.rectangle(grid_mask, (0, 0), (grid_mask.shape[1]-1, grid_mask.shape[0]-1), 255, border_thickness)
    
    free_space = cv2.bitwise_not(grid_mask)
    
    # Run distance transform
    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    max_dist = np.max(dist)
    
    # Seeds
    min_dist = max(15, int(max_dist * 0.5))
    kernel_size = min_dist if min_dist % 2 == 1 else min_dist + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dist_smooth = cv2.GaussianBlur(dist, (5, 5), 0)
    local_max = cv2.dilate(dist_smooth, kernel)
    peaks_mask = (dist_smooth == local_max) & (dist_smooth > 5)
    
    num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
    
    # Run watershed
    markers = np.zeros(free_space.shape, dtype=np.int32)
    for label_id in range(1, num_peaks):
        cx, cy = peak_centroids[label_id]
        px, py = int(round(cx)), int(round(cy))
        markers[py, px] = label_id
        
    bgr_for_watershed = cv2.cvtColor(grid_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr_for_watershed, markers)
    
    # Create slots overlay
    overlay = cv2.imread("d:/pz/output.jpg")
    
    # To draw boundaries, we can find contour of each slot and draw it
    for label_id in range(1, num_peaks):
        mask = (markers == label_id) & (free_space > 0)
        area = np.count_nonzero(mask)
        if area < 1000:
            continue
        # Find contour of mask
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            
    # Save a downscaled version (1024x745)
    h, w = overlay.shape[:2]
    downscaled = cv2.resize(overlay, (1024, int(1024 * h / w)))
    cv2.imwrite("d:/pz/scratch/binary_slots_overlay.jpg", downscaled)
    print(f"Saved visualization overlay to d:/pz/scratch/binary_slots_overlay.jpg")

if __name__ == "__main__":
    visualize()
