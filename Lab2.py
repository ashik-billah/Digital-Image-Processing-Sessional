import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Images/image1.webp',cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error : No image is loaded")
    exit()

# Contrast Stretching
r1,s1 = 70,0
r2,s2 = 140,255
c_img = np.zeros_like(img)
for r in range(256):
    if r<r1:
        c_img[img==r] = (s1/r1) * r
    elif r<r2:
        c_img[img==r] = ((s2-s1) / (r2-r1)) * (r - r1) + s1
    else:
        c_img[img==r] = ((255-s2) / (255-r2)) * (r-r2) + s2
c_img = np.array(c_img,dtype=np.uint8)

# Gray Level Slicing
low,high = 100,180
g_img = np.copy(img)
g_img[(img>=low) & (img<=high)] = 255

plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(c_img,cmap='gray')
plt.title("Contrast Stretched Image")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(g_img,cmap='gray')
plt.title("Gray Level Sliced Image")
plt.axis('off')

plt.show()

