import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('Images/image1.webp',cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error : No image is loaded")
    exit()

row,col = img.shape
total_pixel = row*col
print(f"Row : {row} rows, Column : {col} columns, Total Pixels : {total_pixel}")

hist = [0] * 256
for i in range(row):
    for j in range(col):
        intensity = img[i,j]
        hist[intensity] += 1

pdf = [0.0] * 256
for i in range(256):
    pdf[i] = hist[i]/total_pixel

cdf = [0.0] * 256
cdf[0] = pdf[0]
for i in range(1,256):
    cdf[i] = cdf[i-1] + pdf[i]

eq_map = [0] * 256
for i in range(256):
    eq_map[i] = int(cdf[i] * 255 + 0.5)

eq_img = img.copy()
for i in range(row):
    for j in range(col):
        old_val = img[i,j]
        new_val = eq_map[old_val]
        eq_img[i,j] = new_val

hist_eq = [0]*256
for i in range(row):
    for j in range(col):
        val = eq_img[i,j]
        hist_eq[val] +=1

plt.figure(figsize=(12,6))

plt.subplot(2,2,1)
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(eq_img,cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

plt.subplot(2,2,3)
plt.bar(range(256),hist,color='gray')
plt.title("Original Histogram")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2,2,4)
plt.bar(range(256),hist_eq,color='gray')
plt.title("Equalized Histogram")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

