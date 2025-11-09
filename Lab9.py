import cv2 as cv
from collections import Counter
import heapq
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Step 1: Read grayscale image -----------------
img = cv.imread('Images/image1.webp', cv.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Image not loaded")
    exit()

# ----------------- Step 2: Flatten and get frequencies -----------------
pixels = img.flatten()
freq = Counter(pixels)

# ----------------- Step 3: Build Huffman tree -----------------
heap = [[wt, [sym, ""]] for sym, wt in freq.items()]
heapq.heapify(heap)

while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)
    for pair in lo[1:]:
        pair[1] = '0' + pair[1]
    for pair in hi[1:]:
        pair[1] = '1' + pair[1]
    heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

huff_dict = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

# ----------------- Step 4: Encode -----------------
encoded_data = ''.join(huff_dict[p] for p in pixels)

# ----------------- Step 5: Decode -----------------
decode_dict = {v:k for k,v in huff_dict.items()}
decoded_pixels = []
temp = ""
for bit in encoded_data:
    temp += bit
    if temp in decode_dict:
        decoded_pixels.append(decode_dict[temp])
        temp = ""
decoded_img = np.array(decoded_pixels, dtype=np.uint8).reshape(img.shape)

# ----------------- Step 6: Print Huffman info -----------------
print("Huffman Code Table (partial):")
for k in list(huff_dict.keys())[:10]:
    print(f"{k}: {huff_dict[k]}")
print("\nOriginal size (bits):", len(pixels) * 8)
print("Compressed size (bits):", len(encoded_data))
print("Compression Ratio:", round((len(pixels)*8)/len(encoded_data), 2))

# ----------------- Step 7: Plot Original and Decoded Image -----------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(decoded_img, cmap='gray')
plt.title("Decoded Image")
plt.axis('off')

plt.tight_layout()
plt.show()

