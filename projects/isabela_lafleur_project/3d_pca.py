import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA

from utils import load_data


def pca_3d_plot(save_path="pca_3d.png"):
    X_train, X_val, y_train, y_val, *_ = load_data()

    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        X_train_pca[:, 2], # type: ignore
        c=y_train,
        cmap="tab10",
        s=12,
        alpha=0.7,
        label="train"
    )

    ax.scatter(
        X_val_pca[:, 0],
        X_val_pca[:, 1],
        X_val_pca[:, 2], # type: ignore
        c=y_val,
        cmap="tab10",
        s=12,
        alpha=0.7,
        marker="x",
        label="val"
    )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var[2]*100:.1f}%)")

    ax.set_title("3D PCA of Train/Validation Data")
    ax.legend()

    fig.colorbar(sc, ax=ax, label="Class label")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    pca_3d_plot()
