# src/utils/plot.py: matplotlib helpers for plotting (image, corners) samples

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.data.transforms import Denormalize, ToNumpy

CORNER_COLORS = ["red", "green", "blue", "orange"]
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_OFFSETS = [(-10, -10), (-10, -10), (-10, 20), (-10, 20)]


def show_samples(images, corners=None, ncols=5, title=None, denormalize=False, cell_size=(2, 2)):
    """Plot a grid of (already-transformed) image tensors with optional corner overlay."""
    nrows = (len(images) + ncols - 1) // ncols
    cell_w, cell_h = cell_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * cell_h))
    denorm = Denormalize() if denormalize else None
    to_numpy = ToNumpy()

    for i, ax in enumerate(np.array(axes).flatten()):
        if i >= len(images):
            ax.axis("off")
            continue

        img, pts = images[i], corners[i] if corners is not None else None
        img = denorm(img) if denorm is not None else img
        if pts is not None:
            img, pts = to_numpy(img, pts)
        else:
            img = to_numpy(img)
        h, w = img.shape[:2]

        ax.imshow(img)
        if pts is not None:
            poly_pts = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1)
            ax.add_patch(Polygon(poly_pts, closed=True, fill=False,
                                 edgecolor="white", linewidth=1.2, alpha=0.8))
            for j, (cx, cy) in enumerate(pts):
                px, py = cx * w, cy * h
                dx, dy = CORNER_OFFSETS[j]
                ax.plot(px, py, "o", color=CORNER_COLORS[j], markersize=5)
                ax.text(px + dx, py + dy, CORNER_LABELS[j], color=CORNER_COLORS[j], fontsize=8, weight="bold")

        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def show_history(history, title=None):
    """Plot per-epoch train (and optional valid) curves for each key in history."""
    train = history.get("train", {})
    valid = history.get("valid", {})
    keys = list(train.keys())

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        ax.plot(train[key], lw=2, label="train")
        if key in valid:
            ax.plot(valid[key], lw=2, label="valid")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()

    if title is not None:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()
