import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')


# ======================================================
# LATIHAN 2 : SVM DENGAN BERBAGAI KERNEL
# ======================================================

def praktikum_svm_fruits():

    print("\nLATIHAN 2: SVM DENGAN BERBAGAI KERNEL")
    print("=" * 50)

    # ==================================================
    # MEMBUAT DATASET BUAH
    # ==================================================

    def create_fruit_dataset(n_samples=300):

        np.random.seed(42)
        n_features = 20

        # APPLE
        apples = np.random.randn(n_samples, n_features)
        apples[:, 0] += 2
        apples[:, 1] += 1
        apples_labels = np.zeros(n_samples)

        # BANANA
        bananas = np.random.randn(n_samples, n_features)
        bananas[:, 0] += 1
        bananas[:, 1] += 3
        bananas_labels = np.ones(n_samples)

        # ORANGE
        oranges = np.random.randn(n_samples, n_features)
        oranges[:, 0] += 1.5
        oranges[:, 1] += 1
        oranges[:, 2] += 2
        oranges_labels = np.ones(n_samples) * 2

        X = np.vstack([apples, bananas, oranges])
        y = np.hstack([apples_labels, bananas_labels, oranges_labels])

        return X, y

    X, y = create_fruit_dataset(100)

    fruit_names = ['Apple', 'Banana', 'Orange']

    print(f"Dataset Shape: {X.shape}")
    print(f"Class Distribution: {np.bincount(y.astype(int))}")

    # ==================================================
    # SPLIT DATA
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # ==================================================
    # NORMALISASI DATA
    # ==================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==================================================
    # PCA UNTUK VISUALISASI
    # ==================================================

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X_train_scaled)

    # ==================================================
    # VISUALISASI DATA PCA
    # ==================================================

    plt.figure(figsize=(10, 6))

    colors = ['red', 'yellow', 'orange']

    for i, color in enumerate(colors):

        plt.scatter(
            X_pca[y_train == i, 0],
            X_pca[y_train == i, 1],
            c=color,
            label=fruit_names[i],
            alpha=0.7,
            edgecolors='black'
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Fruit Dataset in PCA Space')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==================================================
    # EKSPERIMEN BERBAGAI KERNEL
    # ==================================================

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.ravel()

    for idx, kernel in enumerate(kernels):

        if kernel == 'poly':

            svm_model = svm.SVC(
                kernel=kernel,
                degree=3,
                C=1.0,
                random_state=42
            )

        else:

            svm_model = svm.SVC(
                kernel=kernel,
                C=1.0,
                random_state=42
            )

        # ==================================================
        # CROSS VALIDATION
        # ==================================================

        cv_scores = cross_val_score(
            svm_model,
            X_train_scaled,
            y_train,
            cv=5
        )

        # ==================================================
        # TRAIN MODEL
        # ==================================================

        svm_model.fit(X_train_scaled, y_train)

        y_pred = svm_model.predict(X_test_scaled)

        accuracy = svm_model.score(X_test_scaled, y_test)

        # ==================================================
        # DECISION BOUNDARY
        # ==================================================

        X0 = X_pca[:, 0]
        X1 = X_pca[:, 1]

        x_min, x_max = X0.min() - 1, X0.max() + 1
        y_min, y_max = X1.min() - 1, X1.max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        # Titik meshgrid dalam PCA space
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        # Kembalikan ke dimensi asli
        mesh_points_original = pca.inverse_transform(mesh_points)

        # Prediksi
        Z = svm_model.predict(mesh_points_original)

        Z = Z.reshape(xx.shape)

        axes[idx].contourf(
            xx,
            yy,
            Z,
            alpha=0.3,
            cmap=plt.cm.RdYlBu
        )

        for i, color in enumerate(colors):

            axes[idx].scatter(
                X_pca[y_train == i, 0],
                X_pca[y_train == i, 1],
                c=color,
                label=fruit_names[i],
                alpha=0.7,
                edgecolors='black'
            )

        axes[idx].set_title(
            f'{kernel.upper()} Kernel\n'
            f'CV Acc: {cv_scores.mean():.3f} | '
            f'Test Acc: {accuracy:.3f}'
        )

        axes[idx].set_xlabel('PC1')
        axes[idx].set_ylabel('PC2')

        results[kernel] = {
            'cv_accuracy': cv_scores.mean(),
            'test_accuracy': accuracy,
            'model': svm_model
        }

    plt.tight_layout()
    plt.show()

    # ==================================================
    # HASIL PERBANDINGAN
    # ==================================================

    print("\nSVM KERNEL COMPARISON")
    print("-" * 45)

    print(f"{'Kernel':<10} {'CV Accuracy':<15} {'Test Accuracy'}")

    print("-" * 45)

    for kernel, result in results.items():

        print(
            f"{kernel:<10} "
            f"{result['cv_accuracy']:<15.4f} "
            f"{result['test_accuracy']:.4f}"
        )

    # ==================================================
    # ROC CURVE
    # ==================================================

    best_kernel = max(
        results,
        key=lambda k: results[k]['test_accuracy']
    )

    print(f"\nBest Kernel: {best_kernel}")

    y_test_bin = label_binarize(
        y_test,
        classes=[0, 1, 2]
    )

    ovr_classifier = OneVsRestClassifier(
        svm.SVC(
            kernel=best_kernel,
            probability=True,
            random_state=42
        )
    )

    ovr_classifier.fit(X_train_scaled, y_train)

    y_score = ovr_classifier.predict_proba(X_test_scaled)

    plt.figure(figsize=(10, 8))

    for i in range(len(fruit_names)):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f'{fruit_names[i]} (AUC = {roc_auc:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(
        f'ROC Curve - SVM {best_kernel.upper()} Kernel'
    )

    plt.legend()
    plt.grid(True)
    plt.show()

    return results, fruit_names


# ======================================================
# MENJALANKAN PROGRAM
# ======================================================

svm_results, fruit_names = praktikum_svm_fruits()