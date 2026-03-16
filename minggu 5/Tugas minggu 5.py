import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

print("Dibuat oleh Rangga Dharma Putra")
print("NIM 24343045")

def calculate_metrics(original, filtered):
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    ssim_value = ssim(original, filtered)

    return mse, psnr, ssim_value


def add_gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image):
    noisy = image.copy()
    prob = 0.02
    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255
    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0
    return noisy


def add_speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise * 0.2
    return np.clip(noisy, 0, 255).astype(np.uint8)


img = cv2.imread("peppers.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image tidak ditemukan")
    exit()

gaussian_noise = add_gaussian_noise(img)
sp_noise = add_salt_pepper_noise(img)
speckle_noise = add_speckle_noise(img)

noise_images = {
    "Gaussian Noise": gaussian_noise,
    "Salt Pepper": sp_noise,
    "Speckle": speckle_noise
}

filters = {
    "Mean 3x3": lambda x: cv2.blur(x, (3,3)),
    "Mean 5x5": lambda x: cv2.blur(x, (5,5)),
    "Gaussian sigma1": lambda x: cv2.GaussianBlur(x, (5,5), 1),
    "Gaussian sigma2": lambda x: cv2.GaussianBlur(x, (5,5), 2),
    "Median 3x3": lambda x: cv2.medianBlur(x, 3),
    "Median 5x5": lambda x: cv2.medianBlur(x, 5),
    "Max Filter": lambda x: cv2.dilate(x, np.ones((3,3),np.uint8))
}

results = []

for noise_name, noisy_img in noise_images.items():
    for filter_name, filter_func in filters.items():
        start = time.time()
        filtered = filter_func(noisy_img)
        end = time.time()

        mse, psnr, ssim_val = calculate_metrics(img, filtered)

        results.append([
            noise_name,
            filter_name,
            mse,
            psnr,
            ssim_val,
            end-start
        ])

print("\nHASIL EVALUASI FILTER\n")

print("{:<15} {:<20} {:<10} {:<10} {:<10} {:<10}".format(
    "Noise","Filter","MSE","PSNR","SSIM","Time"
))

print("-"*80)

for r in results:
    print("{:<15} {:<20} {:<10.2f} {:<10.2f} {:<10.3f} {:<10.4f}".format(
        r[0],r[1],r[2],r[3],r[4],r[5]
    ))

fig, axes = plt.subplots(3,4, figsize=(12,8))

axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")

axes[1].imshow(gaussian_noise, cmap='gray')
axes[1].set_title("Gaussian Noise")

axes[2].imshow(sp_noise, cmap='gray')
axes[2].set_title("Salt Pepper")

axes[3].imshow(speckle_noise, cmap='gray')
axes[3].set_title("Speckle Noise")

axes[4].imshow(cv2.blur(gaussian_noise,(5,5)), cmap='gray')
axes[4].set_title("Mean Filter")

axes[5].imshow(cv2.GaussianBlur(gaussian_noise,(5,5),1), cmap='gray')
axes[5].set_title("Gaussian Filter")

axes[6].imshow(cv2.medianBlur(sp_noise,5), cmap='gray')
axes[6].set_title("Median Filter")

axes[7].imshow(cv2.dilate(speckle_noise,np.ones((3,3),np.uint8)), cmap='gray')
axes[7].set_title("Max Filter")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()