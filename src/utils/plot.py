# src/utils/plot.py: matplotlib helpers for plotting (image, corners) samples

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.data.transforms import Denormalize, ToNumpy

CORNER_COLORS = ["red", "green", "blue", "orange"]
LABELS = ["TL", "TR", "BR", "BL"]
OFFSETS = [(-10, -10), (-10, -10), (-10, 20), (-10, 20)]


def show_samples(images, corners=None, ncols=5, title=None, denormalize=False):
    """Plot a grid of (already-transformed) image tensors with optional corner overlay."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
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
                dx, dy = OFFSETS[j]
                ax.plot(px, py, "o", color=CORNER_COLORS[j], markersize=5)
                ax.text(px + dx, py + dy, LABELS[j], color=CORNER_COLORS[j], fontsize=8, weight="bold")

        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
