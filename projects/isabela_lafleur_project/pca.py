import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import load_data  


def pca_2d_plot(save_path=None):
    X_train, X_val, y_train, y_val, X_test, _ = load_data()

    pca = PCA(n_components=2, random_state=0)
    X_train_2d = pca.fit_transform(X_train)
    X_val_2d = pca.transform(X_val)

    plt.figure(figsize=(7, 5))
    sc1 = plt.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1],
        c=y_train, cmap="tab10", s=12, alpha=0.7, label="train"
    )
    plt.scatter(
        X_val_2d[:, 0], X_val_2d[:, 1],
        c=y_val, cmap="tab10", s=12, alpha=0.7, marker="x", label="val"
    )

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    plt.title("PCA (2D) of Train/Val Data")
    plt.legend(loc="best")

    cbar = plt.colorbar(sc1)
    cbar.set_label("Class label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def pca_explained_variance_plot(max_components=50, save_path=None):
    X_train, *_ = load_data()

    pca = PCA(n_components=min(max_components, X_train.shape[1]), random_state=0)
    pca.fit(X_train)

    cum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum)
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance (train set)")
    plt.ylim(0, 1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    pca_2d_plot(save_path="pca_2d.png")                
    pca_explained_variance_plot(save_path="pca_var.png") 
