import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Images/image1.webp')

if img is None:
    print("Error : No image is loaded")
    exit()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

threshold = 127
b_image = np.zeros_like(gray,dtype=np.uint8)

row,col = gray.shape
for i in range(row):
    for j in range(col):
        if gray[i,j] > threshold:
            b_image[i,j] = 255
        else:
            b_image[i,j] = 0

plt.figure(figsize=(10,8))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(b_image,cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.tight_layout()
plt.show()
