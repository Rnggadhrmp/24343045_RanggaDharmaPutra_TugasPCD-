import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.restoration import richardson_lucy
import time

img = cv2.imread('image.jpg', 0)
if img is None:
    raise Exception("Gambar tidak ditemukan! Pastikan 'image.jpg' ada.")
img = img.astype(np.float64) / 255.0

def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1
    psf /= psf.sum()
    return psf

psf = motion_psf(15, 30)

blur = convolve2d(img, psf, mode='same', boundary='wrap')

gaussian_noise = blur + np.random.normal(0, 20/255.0, img.shape)
gaussian_noise = np.clip(gaussian_noise, 0, 1)

def salt_pepper(image, prob=0.05):
    noisy = image.copy()
    rand = np.random.rand(*image.shape)
    noisy[rand < prob/2] = 0
    noisy[rand > 1 - prob/2] = 1
    return noisy

sp_noise = salt_pepper(blur, 0.05)

def inverse_filter(img, psf):
    eps = 1e-3
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    F_hat = G / (H + eps)
    result = np.abs(np.fft.ifft2(F_hat))
    return np.clip(result, 0, 1)

def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G
    result = np.abs(np.fft.ifft2(F_hat))
    return np.clip(result, 0, 1)

def lucy_richardson_filter(img, psf, iterations=20):
    result = richardson_lucy(img, psf, num_iter=iterations)
    return np.clip(result, 0, 1)

def evaluate(original, restored):
    mse = mean_squared_error(original, restored)
    psnr = peak_signal_noise_ratio(original, restored, data_range=1.0)
    ssim = structural_similarity(original, restored, data_range=1.0)
    return mse, psnr, ssim

def print_eval(name, original, restored, t):
    mse, psnr, ssim = evaluate(original, restored)
    print(f"{name}")
    print(f"  MSE  : {mse:.6f}")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  Time : {t:.4f} sec")
    print("-"*40)

results = {}

datasets = {
    "Blur": blur,
    "Gaussian": gaussian_noise,
    "SaltPepper": sp_noise
}

for key, data in datasets.items():
    print(f"\n=== DATASET: {key} ===")

    start = time.time()
    inv = inverse_filter(data, psf)
    t_inv = time.time() - start

    start = time.time()
    wien = wiener_filter(data, psf, K=0.01)
    t_wien = time.time() - start

    start = time.time()
    lr = lucy_richardson_filter(data, psf, iterations=20)
    t_lr = time.time() - start

    results[key] = {
        "Inverse": inv,
        "Wiener": wien,
        "Lucy-Richardson": lr
    }

    print_eval("Inverse Filter", img, inv, t_inv)
    print_eval("Wiener Filter", img, wien, t_wien)
    print_eval("Lucy-Richardson", img, lr, t_lr)

def show_images(title, images):
    plt.figure(figsize=(12, 6))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(name)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_images("Original & Degradation", {
    "Original": img,
    "Blur": blur,
    "Gaussian": gaussian_noise,
    "SaltPepper": sp_noise
})

for key in results:
    show_images(f"Restoration - {key}", results[key])

def show_fft_combined(images_dict, title="FFT Results"):
    plt.figure(figsize=(12, 4))
    
    for i, (name, image) in enumerate(images_dict.items()):
        f = np.fft.fftshift(np.fft.fft2(image))
        magnitude = np.log(1 + np.abs(f))
        
        plt.subplot(1, len(images_dict), i+1)
        plt.imshow(magnitude, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_fft_combined({
    "Original": img,
    "Blur": blur,
    "Wiener (Gaussian)": results["Gaussian"]["Wiener"]
})

print("\nPipeline selesai dijalankan!")