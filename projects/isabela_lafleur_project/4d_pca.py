import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from utils import load_data


def pca_4d_animation(save_path="pca_4d.gif"):
    X_train, _, y_train, _, *_ = load_data()

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_train)

    pc1, pc2, pc3, pc4 = X_pca.T

    pc4_norm = (pc4 - pc4.min()) / (pc4.max() - pc4.min())

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pc1, pc2, pc3,
        c=y_train,
        cmap="tab10",
        s=12,
        alpha=0.7
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")

    ax.set_title("4D PCA Animation (PC4 as Time)")

    def update(frame):
        offset = (pc4_norm - 0.5) * 4 * np.sin(frame / 5)
        scatter._offsets3d = (pc1, pc2, pc3 + offset) # type: ignore

        ax.view_init(elev=20, azim=frame * 2) #rotates camera to make 4d movement visible
        return scatter,

    anim = FuncAnimation(fig, update, frames=60, interval=100)

    # SAVE AS GIF
    anim.save(save_path, writer="pillow", fps=10)
    print(f"Saved animation to {save_path}")

    plt.show()


if __name__ == "__main__":
    pca_4d_animation()
