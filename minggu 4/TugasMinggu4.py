import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# =========================================
# 1. POINT PROCESSING METHODS
# =========================================

def negative_transform(image):
    return 255 - image


def log_transform(image):

    c = 255 / np.log(1 + np.max(image))
    log_img = c * np.log(1 + image)

    return np.array(log_img, dtype=np.uint8)


def gamma_transform(image, gamma):

    normalized = image / 255.0
    gamma_img = np.power(normalized, gamma)

    return np.uint8(gamma_img * 255)


# =========================================
# 2. CONTRAST STRETCHING
# =========================================

def contrast_stretch_manual(image):

    r_min = np.min(image)
    r_max = np.max(image)

    stretched = (image - r_min) * (255/(r_max-r_min))

    return stretched.astype(np.uint8)


def contrast_stretch_auto(image):

    p2, p98 = np.percentile(image,(2,98))

    stretched = np.clip((image-p2)*255/(p98-p2),0,255)

    return stretched.astype(np.uint8)


# =========================================
# 3. HISTOGRAM EQUALIZATION
# =========================================

def histogram_equalization(image):

    return cv2.equalizeHist(image)


# =========================================
# 4. CLAHE (ADAPTIVE HISTOGRAM EQUALIZATION)
# =========================================

def apply_clahe(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    return clahe.apply(image)


# =========================================
# 5. METRICS
# =========================================

def calculate_entropy(image):

    hist,_ = np.histogram(image.flatten(),256,[0,256])

    hist = hist[hist>0]

    return entropy(hist)


def contrast_ratio(image):

    return (np.max(image)-np.min(image)) / (np.max(image)+np.min(image)+1e-5)


# =========================================
# 6. HISTOGRAM VISUALIZATION
# =========================================

def show_histogram(image,title):

    plt.figure()

    plt.hist(image.ravel(),256,[0,256])

    plt.title(title)

    plt.show()


def compare_histogram(original,enhanced,name):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(original.ravel(),256,[0,256])
    plt.title("Histogram Original")

    plt.subplot(1,2,2)
    plt.hist(enhanced.ravel(),256,[0,256])
    plt.title("Histogram "+name)

    plt.show()


# =========================================
# 7. VISUAL COMPARISON
# =========================================

def show_results(title,results):

    plt.figure(figsize=(14,10))

    for i,(name,img) in enumerate(results.items()):

        plt.subplot(4,3,i+1)

        plt.imshow(img,cmap='gray')

        plt.title(name)

        plt.axis("off")

    plt.suptitle(title)

    plt.show()


# =========================================
# 8. ENHANCEMENT PIPELINE
# =========================================

def enhancement_pipeline(image):

    results = {}

    results["Original"] = image

    # Point processing
    results["Negative"] = negative_transform(image)
    results["Log"] = log_transform(image)
    results["Gamma 0.5"] = gamma_transform(image,0.5)
    results["Gamma 1.0"] = gamma_transform(image,1.0)
    results["Gamma 2.0"] = gamma_transform(image,2.0)

    # Histogram methods
    results["Contrast Stretch (Manual)"] = contrast_stretch_manual(image)
    results["Contrast Stretch (Auto)"] = contrast_stretch_auto(image)
    results["Histogram Equalization"] = histogram_equalization(image)
    results["CLAHE"] = apply_clahe(image)

    return results


# =========================================
# 9. MAIN PROGRAM
# =========================================

if __name__ == "__main__":

    # Load images
    under = cv2.imread("underexposed.jpg",0)
    over = cv2.imread("overexposed.jpg",0)
    uneven = cv2.imread("uneven.png",0)

    if under is None or over is None or uneven is None:
        print("Error: gambar tidak ditemukan")
        exit()

    datasets = {

        "Underexposed":under,
        "Overexposed":over,
        "Uneven Illumination":uneven
    }

    for name,image in datasets.items():

        print("\n=========================")
        print("Processing:",name)

        # jalankan pipeline
        results = enhancement_pipeline(image)

        # tampilkan semua hasil
        show_results(name,results)

        # histogram original
        show_histogram(image,"Histogram Original")

        # histogram CLAHE
        show_histogram(results["CLAHE"],"Histogram CLAHE")

        # perbandingan histogram
        compare_histogram(image,results["CLAHE"],"CLAHE")

        # =====================
        # METRIC EVALUATION
        # =====================

        print("\nMETRIC EVALUATION")

        methods = [
            "Log",
            "Gamma 0.5",
            "Gamma 2.0",
            "Contrast Stretch (Manual)",
            "Histogram Equalization",
            "CLAHE"
        ]

        for m in methods:

            entropy_val = calculate_entropy(results[m])
            contrast_val = contrast_ratio(results[m])

            print("\nMethod:",m)
            print("Entropy :",entropy_val)
            print("Contrast:",contrast_val)