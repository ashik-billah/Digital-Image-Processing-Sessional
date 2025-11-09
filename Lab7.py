import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Read Image -----------------
img = cv.imread('Images/image1.webp', cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not loaded")
    exit()

row, col = img.shape
pad = 1  # for 3x3 masks
padded = np.pad(img.astype(int), ((pad,pad),(pad,pad)), mode='constant', constant_values=0)

# ----------------- Predefined Masks -----------------
mean_mask = [[1,1,1],[1,1,1],[1,1,1]]          # Mean
gauss_mask = [[1,2,1],[2,4,2],[1,2,1]]        # Gaussian
lap_mask = [[0,-1,0],[-1,4,-1],[0,-1,0]]     # Laplacian
sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]        # Sobel X
sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]        # Sobel Y

# ----------------- Mean Filter -----------------
mean_img = np.zeros_like(img)
for i in range(row):
    for j in range(col):
        s = 0
        for x in range(3):
            for y in range(3):
                s += padded[i+x, j+y] * mean_mask[x][y]
        mean_img[i,j] = s // 9  # divide by 9

# ----------------- Gaussian Filter -----------------
gauss_img = np.zeros_like(img)
for i in range(row):
    for j in range(col):
        s = 0
        for x in range(3):
            for y in range(3):
                s += padded[i+x, j+y] * gauss_mask[x][y]
        gauss_img[i,j] = s // 16  # divide by 16

# ----------------- Laplacian Filter -----------------
lap_img = np.zeros_like(img)
for i in range(row):
    for j in range(col):
        s = 0
        for x in range(3):
            for y in range(3):
                s += padded[i+x,j+y] * lap_mask[x][y]
        val = img[i,j] - s
        if val < 0: val = 0
        if val > 255: val = 255
        lap_img[i,j] = val

# ----------------- Sobel Filter -----------------
sobel_img = np.zeros_like(img)
for i in range(row):
    for j in range(col):
        gx = 0
        gy = 0
        for x in range(3):
            for y in range(3):
                gx += padded[i+x,j+y]*sobel_x[x][y]
                gy += padded[i+x,j+y]*sobel_y[x][y]
        val = int((gx**2 + gy**2)**0.5)
        if val>255: val=255
        sobel_img[i,j] = val

# ----------------- Display all results -----------------
titles = ['Original','Mean','Gaussian','Laplacian','Sobel']
images = [img, mean_img, gauss_img, lap_img, sobel_img]

plt.figure(figsize=(12,6))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

