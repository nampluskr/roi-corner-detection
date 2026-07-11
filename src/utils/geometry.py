# src/utils/geometry.py: geometric utilities for corner coordinate manipulation

import numpy as np


def order_corners(corners):
    """Reorder 4 corners into TL, TR, BR, BL order and return (4, 2) np.float32."""
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    tl_idx = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -tl_idx, axis=0)
    return pts.astype(np.float32)


def is_invalid_corners(corners, min_dist=0.02):
    """Return True if any two of the 4 corners are closer than min_dist (normalized)."""
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                return True
    return False


def polygon_area(corners):
    """Shoelace formula: (N, 2) ordered polygon vertices $\to$ float area."""
    pts = np.array(corners, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_to_corners(mask):
    """Binary mask (H, W) $\to$ 4 corner points (4, 2) via x+-y extremes, normalized [0, 1]."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((4, 2), dtype=np.float32)

    tl = (xs[np.argmin(xs + ys)], ys[np.argmin(xs + ys)])
    tr = (xs[np.argmax(xs - ys)], ys[np.argmax(xs - ys)])
    br = (xs[np.argmax(xs + ys)], ys[np.argmax(xs + ys)])
    bl = (xs[np.argmin(xs - ys)], ys[np.argmin(xs - ys)])

    corners = np.array([tl, tr, br, bl], dtype=np.float32) / np.array([w, h], dtype=np.float32)
    return corners.astype(np.float32)
