import cv2
import numpy as np

img = cv2.imread(r'd:\pz\output.jpg', cv2.IMREAD_GRAYSCALE)
print(f'Size: {img.shape[1]}x{img.shape[0]}')
print(f'White (>127): {np.sum(img > 127)}')
print(f'Black (<=127): {np.sum(img <= 127)}')
print(f'Total: {img.size}')
print(f'Unique values count: {len(np.unique(img))}')
vals = np.unique(img)
if len(vals) <= 20:
    print(f'Unique values: {vals}')
else:
    print(f'Min: {vals[0]}, Max: {vals[-1]}')
    print(f'Value distribution: 0={np.sum(img==0)}, 255={np.sum(img==255)}')
