import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv.imread('Images/image1.webp', cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error: No image is loaded")
    exit()

# Negative
neg_img = 255 - img

# Log Transform
img_float = img.astype(np.float32)
log_img = np.log(1 + img_float)
log_img = (log_img / np.max(log_img)) * 255
log_img = np.uint8(log_img)

# Gamma Transform
gamma = 0.5
gamma_img = np.power(img_float / 255.0, gamma)
gamma_img = np.uint8(gamma_img * 255)

# Manual plotting (no array used)
plt.figure(figsize=(12, 8))

# 1st image - Original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image in Grayscale")
plt.axis('off')

# 2nd image - Negative
plt.subplot(2, 2, 2)
plt.imshow(neg_img, cmap='gray')
plt.title("Negative Image")
plt.axis('off')

# 3rd image - Log Transform
plt.subplot(2, 2, 3)
plt.imshow(log_img, cmap='gray')
plt.title("Log Transform")
plt.axis('off')

# 4th image - Gamma Transform
plt.subplot(2, 2, 4)
plt.imshow(gamma_img, cmap='gray')
plt.title(f"Gamma Transform (γ={gamma})")
plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.show()
