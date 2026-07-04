# src/utils/plot.py: matplotlib helpers for plotting (image, corners) samples

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.data.dataset import Dataset

CORNER_COLORS = ["red", "green", "blue", "orange"]


def show_samples(images, corners_list, title, labels=None, n_cols=5, n_rows=4):
    """Plot a grid of images with corner points and a polygon overlay."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(images):
            ax.axis("off")
            continue
        img = np.array(images[i])
        h, w = img.shape[:2]
        pts = np.array(corners_list[i])

        ax.imshow(img)
        for j, (cx, cy) in enumerate(pts):
            px, py = cx * w, cy * h
            ax.plot(px, py, "o", color=CORNER_COLORS[j], markersize=5)
            if labels is not None:
                ax.text(px + 10, py, labels[j], color=CORNER_COLORS[j], fontsize=8, weight="bold")
        poly_x = [pts[k, 0] * w for k in [0, 1, 2, 3, 0]]
        poly_y = [pts[k, 1] * h for k in [0, 1, 2, 3, 0]]
        ax.plot(poly_x, poly_y, "w-", linewidth=1.2, alpha=0.8)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def load_raw_samples(csv_path, num_samples):
    """Randomly sample (image, corners) pairs from a gt_corners.csv-format Dataset."""
    dataset = Dataset(csv_path)
    chosen = random.sample(dataset.samples, num_samples)
    samples = []
    for image_path, corners in chosen:
        image = Image.open(image_path).convert("RGB")
        samples.append((image, corners.copy()))
    return samples


def apply_transform(transform, raw_samples):
    """Apply a single (image, corners) transform to a list of raw samples."""
    images, corners_list = [], []
    for image, corners in raw_samples:
        img_t, corners_t = transform(image, corners.copy())
        images.append(img_t)
        corners_list.append(corners_t)
    return images, corners_list
