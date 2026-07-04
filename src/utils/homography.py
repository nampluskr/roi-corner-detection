# src/utils/homography.py: homography estimation and point reprojection

import numpy as np


def estimate_homography(src_corners, dst_corners):
    """Solve the 3x3 homography mapping src_corners (4,2) to dst_corners (4,2)."""
    src = np.array(src_corners, dtype=np.float64).reshape(4, 2)
    dst = np.array(dst_corners, dtype=np.float64).reshape(4, 2)
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    h = np.append(np.linalg.solve(a, b), 1.0).reshape(3, 3)
    return h


def reproject_points(points, homography):
    """Project (N, 2) points through a 3x3 homography, returning (N, 2) with perspective divide."""
    pts = np.array(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    projected = homogeneous @ homography.T
    return projected[:, :2] / projected[:, 2:3]
