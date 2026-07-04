# src/utils/geometry.py: geometric utilities for corner coordinate manipulation

import numpy as np
import cv2


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


def mask_to_corners(mask):
    """Binary mask (H, W) -> 4 corner points (4, 2) via largest contour, normalized [0, 1]."""
    h, w = mask.shape
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((4, 2), dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    corners = box / np.array([w, h], dtype=np.float32)
    return corners.astype(np.float32)
