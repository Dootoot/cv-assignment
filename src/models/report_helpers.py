import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import cv2

from .helpers import (
    extract_feature_maps,
    construct_superpixel_feature_matrix,
    superpixel_label_from_ground_truth,
)

_PALETTE = ["#c4a882", "#4a7c59"]  # soil, wheat
_POLY_DEGREE = 2


def _load_features(image_folder_path: str, max_images: int = 15):
    images_dir = Path(image_folder_path)
    X_all, y_all = [], []
    count = 0

    for img in sorted(images_dir.glob("*.png")):
        if img.name.endswith("_mask.png") or count >= max_images:
            continue
        image = cv2.imread(str(img))
        binary_mask = cv2.imread(str(img).removesuffix(".png") + "_mask.png", cv2.IMREAD_GRAYSCALE)
        if image is None or binary_mask is None:
            continue
        maps = extract_feature_maps(image)
        X = construct_superpixel_feature_matrix(
            maps["superpixel_labels"], maps["colour_maps"], maps["lbp_map"], maps["sobel_map"]
        )
        y = superpixel_label_from_ground_truth(maps["superpixel_labels"], binary_mask)
        X_all.append(X)
        y_all.append(y)
        count += 1

    return np.vstack(X_all), np.concatenate(y_all).astype(int)


def _build_mesh(X_2d, resolution=400, margin=0.5):
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _render_panel(ax, rf, xx, yy, grid_2d, X_2d, y, title, annotation, transform=None):
    grid_input = transform(grid_2d) if transform is not None else grid_2d
    Z = rf.predict(grid_input).reshape(xx.shape)

    cmap = mcolors.ListedColormap(_PALETTE)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=3.5, zorder=4)
    ax.contour(xx, yy, Z, levels=[0.5], colors="#e63946", linewidths=1.6, zorder=5)

    ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c=_PALETTE[0], edgecolors="k",
               linewidths=0.3, s=10, alpha=0.7, label="Soil", zorder=3)
    ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c=_PALETTE[1], edgecolors="k",
               linewidths=0.3, s=10, alpha=0.7, label="Wheat", zorder=3)

    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.text(
        0.02, 0.02, annotation, transform=ax.transAxes, fontsize=7.5,
        color="#444444", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
    )


def visualise_hyperrectangle_cuts(image_folder_path: str, output_path: str = "report_images/rf_hyperrectangle_intrinsic_vs_inflated.png"):
    X, y = _load_features(image_folder_path)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    poly = PolynomialFeatures(degree=_POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_2d)
    poly_dim = X_poly.shape[1]

    # shallow single tree to make rectangular partitioning visually explicit
    tree_intrinsic = RandomForestClassifier(
        n_estimators=1, max_depth=5, class_weight="balanced", random_state=2006
    )
    tree_inflated = RandomForestClassifier(
        n_estimators=1, max_depth=5, class_weight="balanced", random_state=2006
    )
    tree_intrinsic.fit(X_2d, y)
    tree_inflated.fit(X_poly, y)

    rf_intrinsic = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", min_samples_leaf=3, random_state=2006, n_jobs=-1
    )
    rf_inflated = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", min_samples_leaf=3, random_state=2006, n_jobs=-1
    )
    rf_intrinsic.fit(X_2d, y)
    rf_inflated.fit(X_poly, y)

    xx, yy, grid_2d = _build_mesh(X_2d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        r"Hyperrectangle Cutting: Intrinsic $\mathbb{R}^2$ Space  vs  Degree-2 Polynomial Inflated Space",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # row labels
    for row, label in enumerate(["Single decision tree  (depth ≤ 5)", "Full random forest  (100 trees)"]):
        fig.text(
            0.01, 0.75 - row * 0.5, label,
            fontsize=9, color="#333333", rotation=90,
            va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", alpha=0.6),
        )

    _render_panel(
        axes[0, 0], tree_intrinsic, xx, yy, grid_2d, X_2d, y,
        title=f"Intrinsic $\\mathbb{{R}}^2$  (PCA: {var[0]*100:.1f}% + {var[1]*100:.1f}% var.)",
        annotation="Splits ⊥ PC axes → axis-aligned hyperrectangles",
    )
    _render_panel(
        axes[0, 1], tree_inflated, xx, yy, grid_2d, X_2d, y,
        title=f"Inflated $\\mathbb{{R}}^{{{poly_dim}}}$  (degree-{_POLY_DEGREE} poly of $\\mathbb{{R}}^2$), in $\\mathbb{{R}}^2$",
        annotation=f"Splits ⊥ poly-axes in $\\mathbb{{R}}^{{{poly_dim}}}$ → non-linear boundaries in $\\mathbb{{R}}^2$",
        transform=poly.transform,
    )
    _render_panel(
        axes[1, 0], rf_intrinsic, xx, yy, grid_2d, X_2d, y,
        title=r"Intrinsic $\mathbb{R}^2$",
        annotation=r"Ensemble averages rectangular partitions in $\mathbb{R}^2$",
    )
    _render_panel(
        axes[1, 1], rf_inflated, xx, yy, grid_2d, X_2d, y,
        title=f"Inflated $\\mathbb{{R}}^{{{poly_dim}}}$, visualised in $\\mathbb{{R}}^2$",
        annotation=f"Ensemble averages in $\\mathbb{{R}}^{{{poly_dim}}}$ → smoother curved regions in $\\mathbb{{R}}^2$",
        transform=poly.transform,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")
