import cv2
import numpy as np
import pywt
import time
from matplotlib import pyplot as plt

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

img1 = cv2.imread('natural.jpg', 0)
img2 = cv2.imread('noise_periodic.jpg', 0)

if img1 is None or img2 is None:
    print("Error: Gambar tidak ditemukan")
    exit()

def fft_process(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(np.abs(fshift)+1)
    phase = np.angle(fshift)
    return fshift, magnitude, phase

fshift1, mag1, phase1 = fft_process(img1)
fshift2, mag2, phase2 = fft_process(img2)

coords = np.argwhere(mag2 > np.percentile(mag2, 99))
print("Frekuensi dominan:", coords[:10])

def reconstruct_from_magnitude(magnitude, phase):
    f = magnitude * np.exp(1j * phase)
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

def reconstruct_from_phase(phase):
    f = np.exp(1j * phase)
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

rec_mag = reconstruct_from_magnitude(np.abs(fshift1), phase1)
rec_phase = reconstruct_from_phase(phase1)

def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-rows//2)**2 + (j-cols//2)**2) <= cutoff:
                mask[i,j] = 1
    return mask

def ideal_highpass(shape, cutoff):
    return 1 - ideal_lowpass(shape, cutoff)

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    x = np.linspace(-cols//2, cols//2, cols)
    y = np.linspace(-rows//2, rows//2, rows)
    X, Y = np.meshgrid(x,y)
    return np.exp(-(X**2+Y**2)/(2*(cutoff**2)))

def gaussian_highpass(shape, cutoff):
    return 1 - gaussian_lowpass(shape, cutoff)

def bandpass(shape, low, high):
    return ideal_lowpass(shape, high) * ideal_highpass(shape, low)

def notch_filter(shape, centers, radius):
    mask = np.ones(shape)
    for c in centers:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.sqrt((i-c[0])**2 + (j-c[1])**2) <= radius:
                    mask[i,j] = 0
    return mask

def apply_filter(fshift, mask):
    f = fshift * mask
    return np.abs(np.fft.ifft2(np.fft.ifftshift(f)))

cutoffs = [10, 30, 60]

ilp_results = [apply_filter(fshift1, ideal_lowpass(img1.shape, c)) for c in cutoffs]
ihp_results = [apply_filter(fshift1, ideal_highpass(img1.shape, c)) for c in cutoffs]
glp_results = [apply_filter(fshift1, gaussian_lowpass(img1.shape, c)) for c in cutoffs]
ghp_results = [apply_filter(fshift1, gaussian_highpass(img1.shape, c)) for c in cutoffs]

bp_result = apply_filter(fshift1, bandpass(img1.shape, 10, 50))

rows, cols = img2.shape
centers = [(rows//2+30, cols//2), (rows//2-30, cols//2)]
notch_result = apply_filter(fshift2, notch_filter(img2.shape, centers, 10))

coeffs_haar = pywt.wavedec2(img1, 'haar', level=2)
coeffs_db4 = pywt.wavedec2(img1, 'db4', level=2)

LL, (LH, HL, HH), _ = coeffs_haar

coeffs_mod = list(coeffs_haar)
coeffs_mod[1] = tuple([np.zeros_like(c) for c in coeffs_mod[1]])
rec_wavelet = pywt.waverec2(coeffs_mod, 'haar')

start = time.time()
spatial = cv2.GaussianBlur(img1, (5,5), 0)
time_spatial = time.time() - start

start = time.time()
freq = ilp_results[1]
time_freq = time.time() - start

start = time.time()
freq = apply_filter(fshift1, ideal_lowpass(img1.shape, 30))
time_freq = time.time() - start

psnr_spatial = psnr(img1, spatial)
psnr_freq = psnr(img1, freq)
psnr_wave = psnr(img1, rec_wavelet)

plt.figure(figsize=(14,10))



plt.subplot(3,4,1), plt.imshow(img1, cmap='gray'), plt.title("Original")
plt.subplot(3,4,2), plt.imshow(mag1, cmap='gray'), plt.title("Magnitude")
plt.subplot(3,4,3), plt.imshow(phase1, cmap='gray'), plt.title("Phase")
plt.subplot(3,4,4), plt.imshow(rec_phase, cmap='gray'), plt.title("Reconstruct Phase")

plt.subplot(3,4,5), plt.imshow(ilp_results[0], cmap='gray'), plt.title("ILP cutoff=10")
plt.subplot(3,4,6), plt.imshow(ilp_results[1], cmap='gray'), plt.title("ILP cutoff=30")
plt.subplot(3,4,7), plt.imshow(ilp_results[2], cmap='gray'), plt.title("ILP cutoff=60")
plt.subplot(3,4,8), plt.imshow(ihp_results[1], cmap='gray'), plt.title("IHP")

plt.subplot(3,4,9), plt.imshow(glp_results[1], cmap='gray'), plt.title("Gaussian LP")
plt.subplot(3,4,10), plt.imshow(ghp_results[1], cmap='gray'), plt.title("Gaussian HP")
plt.subplot(3,4,11), plt.imshow(bp_result, cmap='gray'), plt.title("Bandpass")
plt.subplot(3,4,12), plt.imshow(notch_result, cmap='gray'), plt.title("Notch")


plt.tight_layout()
plt.suptitle("Analisis FFT dan Filtering", fontsize=16)
plt.show()

plt.figure(figsize=(8,6))

plt.subplot(2,2,1), plt.imshow(LL, cmap='gray'), plt.title("LL (Approx)")
plt.subplot(2,2,2), plt.imshow(LH, cmap='gray'), plt.title("LH (Horizontal)")
plt.subplot(2,2,3), plt.imshow(HL, cmap='gray'), plt.title("HL (Vertical)")
plt.subplot(2,2,4), plt.imshow(HH, cmap='gray'), plt.title("HH (Diagonal)")

plt.tight_layout()
plt.suptitle("Analisis FFT dan Filtering", fontsize=16)
plt.show()

start = time.time()
rec_wavelet = pywt.waverec2(coeffs_mod, 'haar')
time_wavelet = time.time() - start

print("\n=== PERBANDINGAN METRIK ===")
print("Metode\t\tPSNR\t\tWaktu")
print(f"Spatial\t\t{psnr_spatial:.2f}\t\t{time_spatial:.4f}")
print(f"FFT\t\t{psnr_freq:.2f}\t\t{time_freq:.6f}")
print(f"Wavelet\t\t{psnr_wave:.2f}\t\t{time_wavelet:.4f}")
