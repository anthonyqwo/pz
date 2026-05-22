import cv2
import numpy as np

def analyze():
    img = cv2.imread("d:/pz/output.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load output.jpg")
        return
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    print(f"output.jpg has {num_labels - 1} white connected components.")
    
    # Sort component areas
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    areas.sort(reverse=True)
    print(f"Top 10 component areas: {areas[:10]}")
    
    # Check if there is one huge component (the grid itself)
    large_components = sum(1 for a in areas if a > 10000)
    print(f"Number of components with area > 10000: {large_components}")

if __name__ == "__main__":
    analyze()
