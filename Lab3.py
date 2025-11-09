import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Images/Car2.jpg',cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error : No image is loaded")
    exit()

row,col = img.shape
total_pixel = row *col
print(f"Rows : {row}, Columns : {col}, Total Pixels : {total_pixel}")

h = np.zeros(256,dtype=int)
for i in range(row):
    for j in range(col):
        intensity = img[i,j]
        h[intensity] += 1

prob = h/total_pixel

plt.figure(figsize=(10,6))
plt.subplot(2,3,2)
plt.imshow(img,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,2,3)
plt.bar(range(256),h,color='red',edgecolor='black')
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,255,25))

plt.subplot(2,2,4)
plt.bar(range(256),prob,color='gray',edgecolor='black')
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Probability')
plt.xticks(np.arange(0,255,25))

plt.tight_layout()
plt.show()