import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

from skimage.feature import hog
import cv2

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("KOMPARASI KLASIFIKASI KNN VS SVM UNTUK PENGENALAN OBJEK CITRA")
print("=" * 80)

digits = load_digits()

X_images = digits.images
y = digits.target

indices = np.random.choice(len(X_images), 1000, replace=False)

X_images = X_images[indices]
y = y[indices]

print(f"\nJumlah Dataset : {len(X_images)}")
print(f"Jumlah Kelas   : {len(np.unique(y))}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_images[i], cmap='gray')
    ax.set_title(f"Label : {y[i]}")
    ax.axis('off')

plt.suptitle(" Dataset Digits")
plt.tight_layout()
plt.show()

hog_features = []
hist_features = []

for image in X_images:

    image_uint8 = (image * 16).astype(np.uint8)

    hog_feature = hog(
        image_uint8,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        visualize=False
    )

    hist = cv2.calcHist(
        [image_uint8],
        [0],
        None,
        [16],
        [0, 256]
    ).flatten()

    hist = hist / np.sum(hist)

    hog_features.append(hog_feature)
    hist_features.append(hist)

hog_features = np.array(hog_features)
hist_features = np.array(hist_features)

X = np.hstack([hog_features, hist_features])

print(f"\nDimensi Fitur Gabungan : {X.shape}")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"\nData Train : {X_train.shape}")
print(f"Data Test  : {X_test.shape}")

print("\n" + "=" * 80)
print("EKSPERIMEN KNN")
print("=" * 80)

k_values = [1, 3, 5, 7, 9, 11]
distance_metrics = ['euclidean', 'manhattan', 'minkowski']

knn_results = []

best_knn_accuracy = 0
best_knn_model = None
best_knn_params = None

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for metric in distance_metrics:

    for k in k_values:

        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )

        start_train = time.time()
        knn.fit(X_train, y_train)
        end_train = time.time()

        start_infer = time.time()
        y_pred = knn.predict(X_test)
        end_infer = time.time()

        cv_scores = cross_val_score(
            knn,
            X_scaled,
            y,
            cv=stratified_cv,
            scoring='accuracy'
        )

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average='weighted'
        )

        train_time = end_train - start_train
        inference_time = end_infer - start_infer

        knn_results.append([
            k,
            metric,
            accuracy,
            precision,
            recall,
            f1,
            cv_scores.mean(),
            train_time,
            inference_time
        ])

        print(
            f"K={k:<2} | "
            f"Metric={metric:<10} | "
            f"Accuracy={accuracy:.4f} | "
            f"CV={cv_scores.mean():.4f}"
        )

        if accuracy > best_knn_accuracy:
            best_knn_accuracy = accuracy
            best_knn_model = knn
            best_knn_params = (k, metric)

print("\nBest KNN")
print(f"K              : {best_knn_params[0]}")
print(f"Distance Metric: {best_knn_params[1]}")
print(f"Accuracy       : {best_knn_accuracy:.4f}")

print("\n" + "=" * 80)
print("GRID SEARCH SVM")
print("=" * 80)

svm_param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

svm_grid = GridSearchCV(
    SVC(),
    svm_param_grid,
    cv=stratified_cv,
    scoring='accuracy',
    n_jobs=-1
)

start_grid = time.time()

svm_grid.fit(X_train, y_train)

end_grid = time.time()

best_svm_model = svm_grid.best_estimator_

print(f"\nBest Parameters : {svm_grid.best_params_}")
print(f"Best CV Score   : {svm_grid.best_score_:.4f}")
print(f"Grid Search Time: {end_grid - start_grid:.4f} detik")

start_train = time.time()
best_svm_model.fit(X_train, y_train)
end_train = time.time()

start_infer = time.time()
y_pred_svm = best_svm_model.predict(X_test)
end_infer = time.time()

svm_accuracy = accuracy_score(y_test, y_pred_svm)

svm_precision = precision_score(
    y_test,
    y_pred_svm,
    average='weighted'
)

svm_recall = recall_score(
    y_test,
    y_pred_svm,
    average='weighted'
)

svm_f1 = f1_score(
    y_test,
    y_pred_svm,
    average='weighted'
)

svm_train_time = end_train - start_train
svm_inference_time = end_infer - start_infer

print(f"\nSVM Accuracy  : {svm_accuracy:.4f}")
print(f"SVM Precision : {svm_precision:.4f}")
print(f"SVM Recall    : {svm_recall:.4f}")
print(f"SVM F1-Score  : {svm_f1:.4f}")

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

y_pred_knn = best_knn_model.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0]
)

axes[0].set_title("Confusion Matrix KNN")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Greens',
    ax=axes[1]
)

axes[1].set_title("Confusion Matrix SVM")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

print("\nClassification Report KNN")
print(classification_report(y_test, y_pred_knn))

print("\nClassification Report SVM")
print(classification_report(y_test, y_pred_svm))

print("\n" + "=" * 80)
print("PCA DECISION BOUNDARY")
print("=" * 80)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

svm_pca = SVC(
    kernel='rbf',
    C=svm_grid.best_params_['C'],
    gamma=svm_grid.best_params_['gamma']
)

svm_pca.fit(X_train_pca, y_train_pca)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))

plt.contourf(xx, yy, Z, alpha=0.3)

scatter = plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test_pca,
    edgecolors='black'
)

plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("Decision Boundary SVM dengan PCA 2D")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

print("\n" + "=" * 80)
print("ROC CURVE DAN AUC")
print("=" * 80)

y_test_bin = label_binarize(y_test, classes=np.unique(y))

ovr_classifier = OneVsRestClassifier(
    SVC(
        kernel='rbf',
        probability=True,
        C=svm_grid.best_params_['C'],
        gamma=svm_grid.best_params_['gamma']
    )
)

ovr_classifier.fit(X_train, y_train)

y_score = ovr_classifier.predict_proba(X_test)

plt.figure(figsize=(12, 8))

for i in range(10):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {i} (AUC={roc_auc:.3f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC Curve Multi-Class SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 80)
print("LEARNING CURVE")
print("=" * 80)

train_sizes, train_scores, test_scores = learning_curve(
    best_svm_model,
    X_scaled,
    y,
    cv=stratified_cv,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_mean,
    'o-',
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    test_mean,
    'o-',
    label='Validation Accuracy'
)

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve SVM")
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 80)
print("PERBANDINGAN KNN VS SVM")
print("=" * 80)

print(f"\n{'Metode':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")

print("-" * 60)

print(
    f"{'KNN':<10} "
    f"{best_knn_accuracy:<12.4f} "
    f"{precision_score(y_test, y_pred_knn, average='weighted'):<12.4f} "
    f"{recall_score(y_test, y_pred_knn, average='weighted'):<12.4f} "
    f"{f1_score(y_test, y_pred_knn, average='weighted'):<12.4f}"
)

print(
    f"{'SVM':<10} "
    f"{svm_accuracy:<12.4f} "
    f"{svm_precision:<12.4f} "
    f"{svm_recall:<12.4f} "
    f"{svm_f1:<12.4f}"
)

print("\n" + "=" * 80)
print("WAKTU KOMPUTASI")
print("=" * 80)

print(f"\n{'Model':<10} {'Training Time':<20} {'Inference Time'}")
print("-" * 50)

print(
    f"{'KNN':<10} "
    f"{knn_results[0][7]:<20.6f} "
    f"{knn_results[0][8]:.6f}"
)

print(
    f"{'SVM':<10} "
    f"{svm_train_time:<20.6f} "
    f"{svm_inference_time:.6f}"
)

print("\n" + "=" * 80)
print("ANALISIS")
print("=" * 80)

print("""
1. Nilai k kecil pada KNN cenderung overfitting karena terlalu sensitif terhadap tetangga terdekat.

2. Nilai k besar membuat model lebih stabil namun dapat menyebabkan underfitting.

3. Kernel RBF pada SVM umumnya memberikan akurasi terbaik karena mampu menangani data non-linear.

4. SVM memiliki waktu training lebih lama dibanding KNN, namun inference lebih cepat.

5. Kombinasi fitur HOG dan histogram intensitas memberikan representasi bentuk dan distribusi piksel yang baik.

6. Berdasarkan hasil evaluasi, metode dengan accuracy tertinggi direkomendasikan untuk aplikasi pengenalan objek citra serupa.
""")

print("\nProgram selesai.")