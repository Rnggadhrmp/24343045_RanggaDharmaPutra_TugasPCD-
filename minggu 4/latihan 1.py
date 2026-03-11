import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    equalized_image : hasil citra setelah equalization
    transform : fungsi transformasi
    """

    # 1. Hitung histogram
    histogram = np.zeros(256)

    for pixel in image.flatten():
        histogram[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cdf = np.cumsum(histogram)

    # 3. Hitung transformation function
    cdf_normalized = cdf / cdf[-1]
    transform = np.floor(255 * cdf_normalized).astype('uint8')

    # 4. Apply transformation ke image
    equalized_image = transform[image]

    # 5. Return hasil
    return equalized_image, transform


# =====================
# MAIN PROGRAM
# =====================

# Baca gambar grayscale
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Jalankan histogram equalization manual
equalized_image, transform = manual_histogram_equalization(image)

# Tampilkan gambar
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(2,2,2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis("off")

# Histogram sebelum
plt.subplot(2,2,3)
plt.title("Histogram Original")
plt.hist(image.flatten(), bins=256, range=[0,256])

# Histogram sesudah
plt.subplot(2,2,4)
plt.title("Histogram Equalized")
plt.hist(equalized_image.flatten(), bins=256, range=[0,256])

plt.tight_layout()
plt.show()