from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must be 2D arrays")

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = H - kH + 1
    out_W = W - kW + 1

    out = np.zeros((out_H, out_W), dtype=float)
    for i in range(out_H):
        for j in range(out_W):
            region = image[i : i + kH, j : j + kW]
            out[i, j] = np.sum(region * kernel)
    return out


def to_grayscale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[2] >= 3:
        rgb = x[..., :3].astype(float)
        return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    raise ValueError("Unsupported image shape")


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = plt.imread(path)
    return to_grayscale(img)


def demo(image_path: str | None = None):
    if image_path is None:
        image = np.zeros((28, 28), dtype=float)
        image[10:18, 10:18] = 1.0
    else:
        image = load_image(image_path)
        image = image / (np.max(image) + 1e-12)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    edges_x = convolve2d(image, sobel_x)
    edges_y = convolve2d(image, sobel_y)
    magnitude = np.sqrt(edges_x**2 + edges_y**2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(edges_x, cmap="gray")
    axes[1].set_title("Sobel X")
    axes[2].imshow(magnitude, cmap="gray")
    axes[2].set_title("Edge magnitude")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    demo(img_path)
