# ============================================================
# 1. IMPORT LIBRARY
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

plt.rcParams['figure.figsize'] = (6,4)

print("=== PIPELINE TRANSFORMASI GEOMETRIK UNTUK REGISTRASI CITRA ===")


# ============================================================
# 2. LOAD CITRA (UPLOAD DI COLAB)
# ============================================================

img_ref = cv2.imread("poto_lurus.jpeg")
img_target = cv2.imread("poto_miring.jpeg")

if img_ref is None or img_target is None:
    raise Exception("Pastikan file foto_lurus.jpg dan foto_miring.jpg tersedia.")

img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

h, w, _ = img_target.shape

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Reference Image")
plt.imshow(img_ref)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Target Image")
plt.imshow(img_target)
plt.axis("off")
plt.show()


# ============================================================
# 3. TRANSFORMASI DASAR (TRANSLASI, ROTASI, SCALING, SHEARING)
# ============================================================

print("\n--- TRANSFORMASI DASAR ---")

# Translasi
M_trans = np.float32([[1,0,50],[0,1,30]])
img_trans = cv2.warpAffine(img_target, M_trans, (w,h))

# Rotasi
M_rot = cv2.getRotationMatrix2D((w//2,h//2), 30, 1)
img_rot = cv2.warpAffine(img_target, M_rot, (w,h))

# Scaling
img_scale = cv2.resize(img_target, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
img_scale = img_scale[:h,:w]

# Shearing
M_shear = np.float32([[1,0.3,0],[0.2,1,0]])
img_shear = cv2.warpAffine(img_target, M_shear, (w,h))

plt.figure(figsize=(12,8))
titles = ["Original","Translation","Rotation","Scaling","Shearing"]
images = [img_target,img_trans,img_rot,img_scale,img_shear]

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. KOORDINAT HOMOGEN & MATRIKS KOMPOSIT
# ============================================================

print("\n--- KOORDINAT HOMOGEN ---")

points = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
points_h = np.hstack([points, np.ones((4,1))])

T = np.array([[1,0,2],[0,1,1],[0,0,1]])
R = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4),0],
              [np.sin(np.pi/4),  np.cos(np.pi/4),0],
              [0,0,1]])
S = np.array([[2,0,0],[0,1.5,0],[0,0,1]])

composite = T @ R @ S
result = (composite @ points_h.T).T

print("Matriks Komposit:\n", np.round(composite,3))
print("Hasil Transformasi:\n", np.round(result,3))


# ============================================================
# 5. AFFINE vs PERSPECTIVE
# ============================================================

print("\n--- AFFINE vs PERSPEKTIF ---")

# Affine (3 titik)
pts1_aff = np.float32([[50,50],[200,50],[50,200]])
pts2_aff = np.float32([[10,100],[200,50],[100,250]])

start = time.time()
M_aff = cv2.getAffineTransform(pts1_aff, pts2_aff)
img_aff = cv2.warpAffine(img_target, M_aff, (w,h))
aff_time = time.time() - start

# Perspective (4 titik)
pts1_p = np.float32([[50,50],[w-50,50],[w-50,h-50],[50,h-50]])
pts2_p = np.float32([[0,0],[w,0],[w-100,h],[100,h]])

start = time.time()
M_p = cv2.getPerspectiveTransform(pts1_p, pts2_p)
img_p = cv2.warpPerspective(img_target, M_p, (w,h))
persp_time = time.time() - start

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(img_target); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(img_aff); plt.title("Affine"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(img_p); plt.title("Perspective"); plt.axis("off")
plt.show()

print("Waktu Affine:", aff_time)
print("Waktu Perspective:", persp_time)


# ============================================================
# 6. INTERPOLASI + ERROR MAP + METRIK
# ============================================================

print("\n--- PERBANDINGAN INTERPOLASI ---")

methods = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)

for name, method in methods.items():

    start = time.time()
    result = cv2.warpPerspective(img_target, M_p, (w,h), flags=method)
    comp_time = time.time() - start

    gray_res = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    gray_res = cv2.resize(gray_res, (gray_ref.shape[1], gray_ref.shape[0]))

    mse = mean_squared_error(gray_ref, gray_res)
    psnr = peak_signal_noise_ratio(gray_ref, gray_res)

    error_map = np.abs(gray_ref.astype(float) - gray_res.astype(float))

    print(f"\n{name}")
    print("MSE:", mse)
    print("PSNR:", psnr)
    print("Waktu:", comp_time)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(result)
    plt.title(name)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(error_map, cmap='hot')
    plt.title("Error Map")
    plt.axis("off")
    plt.show()


# ============================================================
# 7. IMAGE REGISTRATION (ORB + RANSAC)
# ============================================================

print("\n--- REGISTRASI OTOMATIS ---")

ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(ref_gray, None)
kp2, des2 = orb.detectAndCompute(tar_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:50]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

M_est, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)

if M_est is not None:
    registered = cv2.warpAffine(img_target, M_est, (img_ref.shape[1], img_ref.shape[0]))
else:
    registered = img_target.copy()

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(img_ref); plt.title("Reference"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(img_target); plt.title("Target"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(registered); plt.title("Registered"); plt.axis("off")
plt.show()

print("\nPipeline selesai.")