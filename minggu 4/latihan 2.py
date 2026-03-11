import cv2
import numpy as np


def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    
    Parameters:
    medical_image : input medical image
    modality : 'X-ray', 'MRI', 'CT', 'Ultrasound'
    
    Returns:
    enhanced_image
    enhancement_report
    """

    report = {}

    # ubah ke grayscale jika gambar berwarna
    if len(medical_image.shape) == 3:
        medical_image = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)

    # Enhancement berdasarkan modality
    if modality == 'X-ray':
        enhanced = cv2.equalizeHist(medical_image)
        report["method"] = "Histogram Equalization"

    elif modality == 'MRI':
        enhanced = cv2.GaussianBlur(medical_image, (5,5), 0)
        report["method"] = "Gaussian Blur (Noise Reduction)"

    elif modality == 'CT':
        enhanced = cv2.normalize(medical_image, None, 0, 255, cv2.NORM_MINMAX)
        report["method"] = "Contrast Stretching"

    elif modality == 'Ultrasound':
        enhanced = cv2.medianBlur(medical_image, 5)
        report["method"] = "Median Filter (Speckle Noise Reduction)"

    else:
        enhanced = medical_image
        report["method"] = "No Enhancement"

    # Hitung metrics sederhana
    report["mean_intensity"] = float(np.mean(enhanced))
    report["std_intensity"] = float(np.std(enhanced))

    return enhanced, report


# =====================
# MAIN PROGRAM
# =====================

# Baca gambar
image = cv2.imread("medical.jpg")

# Jalankan enhancement
enhanced_image, report = medical_image_enhancement(image, modality='X-ray')

# Tampilkan gambar
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", enhanced_image)

print("Enhancement Report:")
for key, value in report.items():
    print(key, ":", value)

cv2.waitKey(0)
cv2.destroyAllWindows()