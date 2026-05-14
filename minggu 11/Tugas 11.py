import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations

def load_dataset(path):

    images = []
    labels = []
    class_names = sorted(os.listdir(path))

    for label, cls in enumerate(class_names):

        cls_path = os.path.join(path, cls)

        for file in os.listdir(cls_path):

            img = cv2.imread(
                os.path.join(cls_path, file)
            )

            if img is None:
                continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            blur = cv2.GaussianBlur(
                gray,
                (7,7),
                0
            )

            _, thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            thresh = cv2.bitwise_not(thresh)

            kernel = np.ones((5,5), np.uint8)

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_OPEN,
                kernel
            )

            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_CLOSE,
                kernel
            )

            thresh = cv2.erode(
                thresh,
                kernel,
                iterations=1
            )

            thresh = cv2.dilate(
                thresh,
                kernel,
                iterations=1
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if len(contours) == 0:
                continue

            contour = max(
                contours,
                key=cv2.contourArea
            )

            mask = np.zeros_like(thresh)

            cv2.drawContours(
                mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

            images.append(mask)
            labels.append(label)

    return images, labels, class_names

def region_features(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    M = cv2.moments(contour)

    cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h if h != 0 else 0

    rect_area = w * h

    extent = area / rect_area if rect_area != 0 else 0

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area != 0 else 0

    return [
        area,
        perimeter,
        cx,
        cy,
        w,
        h,
        hull_area,
        aspect_ratio,
        extent,
        solidity
    ]

def moment_features(contour):

    M = cv2.moments(contour)

    spatial = [
        M["m00"],
        M["m10"],
        M["m01"]
    ]

    central = [
        M["mu20"],
        M["mu02"],
        M["mu11"]
    ]

    hu = cv2.HuMoments(M).flatten()

    hu = -np.sign(hu) * np.log10(
        np.abs(hu) + 1e-10
    )

    return spatial + central + list(hu)

def chain_code_8(contour):

    points = contour.squeeze()

    dirs = [
        (1,0),
        (1,1),
        (0,1),
        (-1,1),
        (-1,0),
        (-1,-1),
        (0,-1),
        (1,-1)
    ]

    codes = []

    for i in range(len(points)-1):

        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]

        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

        for idx, (x, y) in enumerate(dirs):

            if dx == x and dy == y:
                codes.append(idx)

    return codes[:30]

def chain_code_4(contour):

    points = contour.squeeze()

    dirs = [
        (1,0),
        (0,1),
        (-1,0),
        (0,-1)
    ]

    codes = []

    for i in range(len(points)-1):

        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]

        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

        if abs(dx) > abs(dy):
            dy = 0
        else:
            dx = 0

        for idx, (x, y) in enumerate(dirs):

            if dx == x and dy == y:
                codes.append(idx)

    return codes[:30]

def normalize_chain_code_8(codes):

    if len(codes) == 0:
        return [0] * 30

    diff = []

    for i in range(len(codes)):
        diff.append(
            (codes[i] - codes[i-1]) % 8
        )

    diff += [0] * (30 - len(diff))

    return diff[:30]

def normalize_chain_code_4(codes):

    if len(codes) == 0:
        return [0] * 30

    diff = []

    for i in range(len(codes)):
        diff.append(
            (codes[i] - codes[i-1]) % 4
        )

    diff += [0] * (30 - len(diff))

    return diff[:30]

def polygon_features(contour):

    epsilon = 0.01 * cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return [len(approx)]

def fourier_features(contour, n=20):

    pts = contour.squeeze()

    complex_pts = pts[:,0] + 1j * pts[:,1]

    complex_pts = complex_pts - np.mean(complex_pts)

    fd = np.fft.fft(complex_pts)

    fd = np.abs(fd)

    fd = fd[1:]

    if fd[0] != 0:
        fd = fd / fd[0]

    if len(fd) < n:
        fd = np.pad(
            fd,
            (0, n - len(fd))
        )

    return list(fd[:n])

def reconstruct_fourier(contour, descriptors):

    pts = contour.squeeze()

    complex_pts = pts[:,0] + 1j * pts[:,1]

    complex_pts = complex_pts - np.mean(complex_pts)

    fd = np.fft.fft(complex_pts)

    truncated = np.zeros_like(fd)

    truncated[:descriptors] = fd[:descriptors]
    truncated[-descriptors:] = fd[-descriptors:]

    reconstructed = np.fft.ifft(truncated)

    return reconstructed

def extract_features(images):

    features = []

    for img in images:

        contours, _ = cv2.findContours(
            img,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        f1 = region_features(contour)

        f2 = moment_features(contour)

        cc8 = chain_code_8(contour)
        cc8 = normalize_chain_code_8(cc8)

        cc4 = chain_code_4(contour)
        cc4 = normalize_chain_code_4(cc4)

        f3 = polygon_features(contour)

        f4 = fourier_features(contour)

        feature_vector = (
            f1 +
            f2 +
            cc8 +
            cc4 +
            f3 +
            f4
        )

        features.append(feature_vector)

    return np.array(features)

def evaluate_feature_sets(X, y):

    feature_sets = {
        "Region": X[:, :10],
        "Moments": X[:, 10:23],
        "ChainCode": X[:, 23:83],
        "Polygon": X[:, 83:84],
        "Fourier": X[:, 84:]
    }

    print("\n=== Accuracy Tiap Kategori ===")

    accuracy_results = {}

    for name, data in feature_sets.items():

        X_train, X_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=0.3,
            random_state=42
        )

        knn = KNeighborsClassifier(
            n_neighbors=3
        )

        knn.fit(
            X_train,
            y_train
        )

        y_pred = knn.predict(X_test)

        acc = accuracy_score(
            y_test,
            y_pred
        )

        accuracy_results[name] = acc

        print(name, "Accuracy :", acc)

    print("\n=== Accuracy Kombinasi Feature ===")

    keys = list(feature_sets.keys())

    for comb in combinations(keys, 2):

        data = np.hstack((
            feature_sets[comb[0]],
            feature_sets[comb[1]]
        ))

        X_train, X_test, y_train, y_test = train_test_split(
            data,
            y,
            test_size=0.3,
            random_state=42
        )

        knn = KNeighborsClassifier(
            n_neighbors=3
        )

        knn.fit(
            X_train,
            y_train
        )

        y_pred = knn.predict(X_test)

        acc = accuracy_score(
            y_test,
            y_pred
        )

        print(comb, ":", acc)

    best_feature = max(
        accuracy_results,
        key=accuracy_results.get
    )

    print("\nFitur Paling Diskriminatif :", best_feature)

def visualize_per_class(images, labels, class_names):

    total_classes = len(class_names)

    for cls_idx in range(total_classes):

        class_images = []

        for i in range(len(images)):

            if labels[i] == cls_idx:
                class_images.append(images[i])

        plt.figure(figsize=(18,12))

        for j in range(len(class_images)):

            img = class_images[j]

            contours, _ = cv2.findContours(
                img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if len(contours) == 0:
                continue

            contour = max(
                contours,
                key=cv2.contourArea
            )

            canvas = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

            hull = cv2.convexHull(contour)

            epsilon = 0.01 * cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            x, y, w, h = cv2.boundingRect(contour)

            cv2.drawContours(
                canvas,
                [contour],
                -1,
                (0,255,0),
                2
            )

            cv2.drawContours(
                canvas,
                [hull],
                -1,
                (255,0,0),
                2
            )

            cv2.drawContours(
                canvas,
                [approx],
                -1,
                (0,0,255),
                2
            )

            cv2.rectangle(
                canvas,
                (x,y),
                (x+w,y+h),
                (255,255,0),
                2
            )

            plt.subplot(4,6,j+1)

            plt.imshow(canvas)

            plt.title(
                f"{class_names[cls_idx]}-{j+1}"
            )

            plt.axis("off")

            recon5 = reconstruct_fourier(
                contour,
                5
            )

            plt.subplot(4,6,j+7)

            plt.plot(
                recon5.real,
                recon5.imag
            )

            plt.title("FD-5")

            plt.axis("equal")

            plt.axis("off")

            recon10 = reconstruct_fourier(
                contour,
                10
            )

            plt.subplot(4,6,j+13)

            plt.plot(
                recon10.real,
                recon10.imag
            )

            plt.title("FD-10")

            plt.axis("equal")

            plt.axis("off")

            recon20 = reconstruct_fourier(
                contour,
                20
            )

            plt.subplot(4,6,j+19)

            plt.plot(
                recon20.real,
                recon20.imag
            )

            plt.title("FD-20")

            plt.axis("equal")

            plt.axis("off")

        plt.tight_layout()

        plt.show()

def run_pipeline(dataset_path):

    images, labels, class_names = load_dataset(
        dataset_path
    )

    print("Jumlah Data :", len(images))

    print("Class :", class_names)

    visualize_per_class(
        images,
        labels,
        class_names
    )

    X = extract_features(images)

    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    knn = KNeighborsClassifier(
        n_neighbors=3
    )

    knn.fit(
        X_train,
        y_train
    )

    y_pred = knn.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    print("\nOverall Accuracy :", acc)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix :")

    print(cm)

    evaluate_feature_sets(
        X,
        y
    )

if __name__ == "__main__":

    dataset_path = "dataset"

    run_pipeline(dataset_path)