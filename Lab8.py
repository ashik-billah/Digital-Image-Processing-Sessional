import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Read image (grayscale) --
img = cv2.imread('Images/image1.webp', cv2.IMREAD_GRAYSCALE)
if img is None:
    exit()

# --- Convert to binary manually (no cv2.threshold) ---
thresh_val = 128
# Create binary image A: 255 for foreground, 0 for background
A = np.zeros_like(img, dtype=np.uint8)
A[img > thresh_val] = 255
A[img <= thresh_val] = 0

# --- Define structuring element B (3x3 all-ones) ---
B = np.array([[1,1,1],
              [1,1,1],
              [1,1,1]], dtype=np.uint8)
bh, bw = B.shape
pad_h, pad_w = bh//2, bw//2

# --- Pad A to handle borders (manual erosion needs neighbors) ---
padded = np.pad(A, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

# --- Manual erosion (no cv2.erode) ---
h, w = A.shape
A_eroded = np.zeros_like(A, dtype=np.uint8)

# For each pixel center in original image coordinate, check if B fits entirely
for i in range(h):
    for j in range(w):
        # region in padded corresponding to center (i,j)
        region = padded[i : i + bh, j : j + bw]
        # check positions where B==1 are all 255 in region
        # (equivalent to checking B shifted fits inside the object)
        # Use numpy operations (no cv2 functions)
        if np.all(region[B == 1] == 255):
            A_eroded[i, j] = 255
        else:
            A_eroded[i, j] = 0

# --- Morphological boundary: β(A) = A - (A ⊖ B) ---
# Implement subtraction manually without cv2.subtract:
# boundary pixel = 255 if A==255 and A_eroded==0, else 0
boundary = np.zeros_like(A, dtype=np.uint8)
boundary[(A == 255) & (A_eroded == 0)] = 255

# --- Display original binary and boundary side by side ---
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original (binary)")
plt.imshow(A, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Morphological Boundary β(A)")
plt.imshow(boundary, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()

