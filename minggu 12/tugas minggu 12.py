import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# DATASET CONFIGURATION
# =========================================================

dataset_path = "dataset"

classes = ["buku", "mug", "botol", "mainan", "remote"]

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(path):

    image_paths = []
    labels = []

    for label, cls in enumerate(classes):

        cls_path = os.path.join(path, cls)

        if not os.path.exists(cls_path):
            continue

        for file in sorted(os.listdir(cls_path)):

            if file.lower().endswith((".jpg", ".png", ".jpeg")):

                full_path = os.path.join(cls_path, file)

                image_paths.append(full_path)
                labels.append(label)

    return image_paths, np.array(labels)

# =========================================================
# FEATURE EXTRACTOR
# =========================================================

def get_feature_extractor(method):

    if method == "SIFT":
        return cv2.SIFT_create()

    elif method == "SURF":
        try:
            return cv2.xfeatures2d.SURF_create(400)
        except:
            return None

    elif method == "ORB":
        return cv2.ORB_create(nfeatures=1000)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image_paths, method):

    extractor = get_feature_extractor(method)

    descriptors_list = []
    keypoints_info = []

    total_time = 0

    for path in image_paths:

        img = cv2.imread(path, 0)

        start = time.time()

        kp, des = extractor.detectAndCompute(img, None)

        end = time.time()

        total_time += (end - start)

        if des is None:
            if method == "ORB":
                des = np.zeros((1, 32), dtype=np.uint8)
            else:
                des = np.zeros((1, 128), dtype=np.float32)

        descriptors_list.append(des)

        keypoints_info.append(len(kp))

    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0
    descriptor_dim = descriptors_list[0].shape[1] if len(descriptors_list) > 0 else 0

    return descriptors_list, keypoints_info, avg_time, descriptor_dim

# =========================================================
# VISUALIZE KEYPOINTS
# =========================================================

def visualize_keypoints(image_path, method):

    extractor = get_feature_extractor(method)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = extractor.detectAndCompute(gray, None)

    result = cv2.drawKeypoints(
        img,
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return result

# =========================================================
# SAMPLE INDICES PER CLASS
# =========================================================

def get_sample_indices_per_class(labels):

    sample_indices = []

    for class_idx in range(len(classes)):

        indices = np.where(labels == class_idx)[0]

        if len(indices) > 0:
            sample_indices.append(indices[0])

    return sample_indices

# =========================================================
# BRUTE FORCE MATCHING
# =========================================================

def brute_force_matching(des1, des2, method):

    if method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return good_matches

# =========================================================
# FLANN MATCHING
# =========================================================

def flann_matching(des1, des2, method):

    if method == "ORB":

        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )

        search_params = dict(checks=50)

    else:

        index_params = dict(
            algorithm=1,
            trees=5
        )

        search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if method == "ORB":
        des1 = np.uint8(des1)
        des2 = np.uint8(des2)
    else:
        des1 = np.float32(des1)
        des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return good_matches

# =========================================================
# RANSAC HOMOGRAPHY
# =========================================================

def apply_ransac(kp1, kp2, matches):

    if len(matches) < 4:
        return None, None

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    return H, mask

# =========================================================
# BOVW
# =========================================================

def build_bovw(descriptors_list, k=20):

    all_descriptors = []

    for des in descriptors_list:

        if des is not None:

            if des.dtype != np.float32:
                des = np.float32(des)

            all_descriptors.extend(des)

    all_descriptors = np.array(all_descriptors)

    if len(all_descriptors) == 0:
        return None

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(all_descriptors)

    return kmeans

# =========================================================
# HISTOGRAM REPRESENTATION
# =========================================================

def create_histogram(descriptors, kmeans, k):

    histogram = np.zeros(k)

    if descriptors is None or kmeans is None:
        return histogram

    if descriptors.dtype != np.float32:
        descriptors = np.float32(descriptors)

    predictions = kmeans.predict(descriptors)

    for p in predictions:
        histogram[p] += 1

    return histogram

# =========================================================
# CREATE ALL HISTOGRAMS
# =========================================================

def create_bovw_features(descriptors_list, kmeans, k):

    features = []

    for des in descriptors_list:

        hist = create_histogram(des, kmeans, k)

        features.append(hist)

    return np.array(features)

# =========================================================
# PCA REDUCTION
# =========================================================

def apply_pca(features, n_components):

    pca = PCA(n_components=n_components)

    reduced = pca.fit_transform(features)

    return reduced, pca

# =========================================================
# CLASSIFICATION
# =========================================================

def classification(features, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(probability=True)

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nSVM Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

    return svm, X_test, y_test, cm

# =========================================================
# PRECISION RECALL CURVE
# =========================================================

def precision_recall_analysis(model, X_test, y_test):

    y_score = model.predict_proba(X_test)

    y_true = (y_test == y_test[0]).astype(int)

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score[:, 0]
    )

    ap = average_precision_score(y_true, y_score[:, 0])

    return precision, recall, ap

# =========================================================
# COMBINED VISUALIZATION
# =========================================================

def show_combined_visualization(
    method,
    visual_images,
    visual_titles,
    matched_image,
    cm,
    precision,
    recall,
    ap
):

    fig = plt.figure(figsize=(18, 12))

    positions = [1, 2, 3, 4, 5]

    for i, img in enumerate(visual_images[:5]):

        ax = plt.subplot(3, 3, positions[i])

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        title = visual_titles[i] if i < len(visual_titles) else f"Class {i + 1}"
        ax.set_title(f"{title} - {method}")

        ax.axis("off")

    ax_match = plt.subplot(3, 3, 6)

    ax_match.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    ax_match.set_title("Feature Matching")
    ax_match.axis("off")

    ax_cm = plt.subplot(3, 3, 7)

    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Confusion Matrix")

    ax_cm.set_xticks(np.arange(len(classes)))
    ax_cm.set_yticks(np.arange(len(classes)))

    ax_cm.set_xticklabels(classes, rotation=45, ha="right")
    ax_cm.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    ax_pr = plt.subplot(3, 3, 8)

    ax_pr.plot(recall, precision)
    ax_pr.set_title(f"Precision Recall Curve AP={ap:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True)

    ax_empty = plt.subplot(3, 3, 9)
    ax_empty.axis("off")

    plt.tight_layout()
    plt.show()

# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    image_paths, labels = load_dataset(dataset_path)

    methods = ["SIFT", "SURF", "ORB"]

    for method in methods:

        print("\n================================================")
        print("METHOD:", method)
        print("================================================")

        extractor = get_feature_extractor(method)

        if extractor is None:

            print("SURF tidak tersedia pada OpenCV ini")
            continue

        descriptors_list, kp_info, avg_time, desc_dim = extract_features(
            image_paths,
            method
        )

        print("Jumlah gambar :", len(image_paths))
        print("Rata-rata keypoints :", np.mean(kp_info))
        print("Waktu ekstraksi :", avg_time)
        print("Dimensi descriptor :", desc_dim)

        sample_indices = get_sample_indices_per_class(labels)
        visual_images = []
        visual_titles = []

        for idx in sample_indices:
            visual_images.append(visualize_keypoints(image_paths[idx], method))
            visual_titles.append(classes[labels[idx]])

        img1 = cv2.imread(image_paths[0])
        img2 = cv2.imread(image_paths[1])

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = extractor.detectAndCompute(img1_gray, None)
        kp2, des2 = extractor.detectAndCompute(img2_gray, None)

        bf_matches = brute_force_matching(des1, des2, method)

        print("BF Good Matches :", len(bf_matches))

        flann_matches = flann_matching(des1, des2, method)

        print("FLANN Good Matches :", len(flann_matches))

        H, mask = apply_ransac(kp1, kp2, bf_matches)

        if mask is not None:

            inliers = np.sum(mask)
            outliers = len(mask) - inliers

            print("RANSAC Inliers :", inliers)
            print("RANSAC Outliers :", outliers)

            matched_img = cv2.drawMatches(
                img1,
                kp1,
                img2,
                kp2,
                bf_matches,
                None,
                flags=2
            )

        else:

            matched_img = np.hstack((img1, img2))

        for k in [10, 20, 50, 100]:

            print("\nVocabulary Size:", k)

            kmeans = build_bovw(descriptors_list, k)

            if kmeans is None:
                print("KMeans dilewati karena descriptor kosong")
                continue

            features = create_bovw_features(
                descriptors_list,
                kmeans,
                k
            )

            max_pca = min(features.shape[0], features.shape[1])

            for n_comp in [16, 32, 64, 128]:

                if n_comp > max_pca:

                    print(f"PCA {n_comp} dilewati karena jumlah data hanya {max_pca}")

                    continue

                reduced, pca = apply_pca(
                    features,
                    n_comp
                )

                print("PCA Components:", n_comp)

                model, X_test, y_test, cm = classification(
                    reduced,
                    labels
                )

                precision, recall, ap = precision_recall_analysis(
                    model,
                    X_test,
                    y_test
                )

                show_combined_visualization(
                    method,
                    visual_images,
                    visual_titles,
                    matched_img,
                    cm,
                    precision,
                    recall,
                    ap
                )

    print("\n================================================")
    print("PROGRAM SELESAI")
    print("================================================")