import sys
sys.path.insert(0, "d:/pz/src")
import cv2
import numpy as np

def inspect():
    img_path = "d:/pz/output.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load d:/pz/output.jpg")
        return
    print(f"output.jpg shape: {img.shape}")
    
    # Check if find_board_corners works on it
    from puzzle_recognition.auto_board_importer import find_board_corners
    try:
        corners = find_board_corners(img)
        print("find_board_corners succeeded!")
        print(f"Corners: {corners.tolist()}")
    except Exception as e:
        print(f"find_board_corners failed: {e}")

if __name__ == "__main__":
    inspect()
