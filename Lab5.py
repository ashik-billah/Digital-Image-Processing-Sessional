import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread('Images/image1.webp')
img2 = cv.imread('Images/image1.webp')

if img1 is None or img2 is None:
    print("Error : No images are loaded")
    exit()

img2 = cv.resize(img2,(img1.shape[1], img1.shape[0]))

img1_f = img1.astype(np.float32)
img2_f = img2.astype(np.float32)

row, col, ch = img1.shape

add = np.zeros_like(img1)
sub = np.zeros_like(img1)
mul = np.zeros_like(img1)
div = np.zeros_like(img1)
avg = np.zeros_like(img1)

# Logical operations arrays
and_img = np.zeros_like(img1)
or_img  = np.zeros_like(img1)
xor_img = np.zeros_like(img1)
not_img1 = np.zeros_like(img1)
not_img2 = np.zeros_like(img2)

for i in range(row):
    for j in range(col):
        for k in range(ch):
            # ---------------- Arithmetic Operations ----------------
            val = img1_f[i,j,k] + img2_f[i,j,k]
            add[i,j,k] = 255 if val>255 else int(val)

            val = img1_f[i,j,k] - img2_f[i,j,k]
            sub[i,j,k] = 0 if val<0 else int(val)

            val = (img1_f[i,j,k] * img2_f[i,j,k]) / 255.0
            mul[i,j,k] = 255 if val>255 else int(val)

            val = (img1_f[i,j,k]) / (img2_f[i,j,k] + 1e-5)
            val = val if val<255 else 255
            div[i,j,k] = int(val)

            val = (img1_f[i,j,k] + img2_f[i,j,k]) / 2.0
            avg[i,j,k] = int(val)

            # ---------------- Logical Operations (manual) ----------------
            # Convert values to integer
            a = int(img1[i,j,k])
            b = int(img2[i,j,k])

            # AND
            c = 0
            for bit in range(8):
                c |= (( (a>>bit)&1 ) & ( (b>>bit)&1 )) << bit
            and_img[i,j,k] = c

            # OR
            c = 0
            for bit in range(8):
                c |= (( (a>>bit)&1 ) | ( (b>>bit)&1 )) << bit
            or_img[i,j,k] = c

            # XOR
            c = 0
            for bit in range(8):
                c |= (( (a>>bit)&1 ) ^ ( (b>>bit)&1 )) << bit
            xor_img[i,j,k] = c

            # NOT
            not_img1[i,j,k] = 255 - a
            not_img2[i,j,k] = 255 - b

# Convert all images to RGB for matplotlib
img1_rgb = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
add_rgb = cv.cvtColor(add,cv.COLOR_BGR2RGB)
sub_rgb = cv.cvtColor(sub,cv.COLOR_BGR2RGB)
mul_rgb = cv.cvtColor(mul,cv.COLOR_BGR2RGB)
div_rgb = cv.cvtColor(div,cv.COLOR_BGR2RGB)
avg_rgb = cv.cvtColor(avg,cv.COLOR_BGR2RGB)

and_rgb = cv.cvtColor(and_img, cv.COLOR_BGR2RGB)
or_rgb  = cv.cvtColor(or_                                                                                                                                        img, cv.COLOR_BGR2RGB)
xor_rgb = cv.cvtColor(xor_img, cv.COLOR_BGR2RGB)
not1_rgb = cv.cvtColor(not_img1, cv.COLOR_BGR2RGB)
not2_rgb = cv.cvtColor(not_img2, cv.COLOR_BGR2RGB)

# Prepare images and titles for plotting
images = [(img1_rgb,"Original Image 1"),
          (img2_rgb,"Original Image 2"),
          (add_rgb,"Added Image"),
          (sub_rgb,"Subtracted Image"),
          (mul_rgb,"Multiplied Image"),
          (div_rgb,"Divided Image"),
          (avg_rgb,"Average Image"),
          (and_rgb,"AND Image"),
          (or_rgb,"OR Image"),
          (xor_rgb,"XOR Image"),
          (not1_rgb,"NOT Image 1"),
          (not2_rgb,"NOT Image 2")]

plt.figure(figsize=(22,15))  # larger figure for 12 images

for i,(img,title) in enumerate(images,1):
    plt.subplot(3,4,i)
    plt.imshow(img)
    plt.title(title, fontsize=12)
    plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.3)  # horizontal & vertical spacing
plt.show()


