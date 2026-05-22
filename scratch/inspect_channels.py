import cv2
import numpy as np

def check():
    img = cv2.imread("d:/pz/output.jpg")
    if img is None:
        return
    
    # Check if grayscale or color
    b, g, r = cv2.split(img)
    diff_rg = np.max(np.abs(r.astype(int) - g.astype(int)))
    diff_gb = np.max(np.abs(g.astype(int) - b.astype(int)))
    
    print(f"Max color channel difference (R-G): {diff_rg}, (G-B): {diff_gb}")
    if diff_rg < 5 and diff_gb < 5:
        print("Image is basically grayscale.")
    else:
        print("Image is color.")
        
    print(f"Mean intensity: {np.mean(img):.2f}")
    print(f"Min intensity: {np.min(img)}, Max intensity: {np.max(img)}")
    
    # Calculate a histogram of intensities
    hist = cv2.calcHist([img], [0], None, [10], [0, 256])
    print("Histogram (10 bins):")
    for i, count in enumerate(hist):
        print(f"  Bin {i*25.6:.1f}-{(i+1)*25.6:.1f}: {int(count[0])}")

if __name__ == "__main__":
    check()
