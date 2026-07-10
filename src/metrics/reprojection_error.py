# src/metrics/reprojection_error.py: homography reprojection error between predicted and ground-truth corners

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.utils.homography import estimate_homography, reproject_points
from src.utils.geometry import is_invalid_corners

DEFAULT_REF_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)


def make_grid(grid_size):
    """Return a (grid_size*grid_size, 2) array of points on the unit square [0, 1]^2."""
    axis = np.linspace(0.0, 1.0, grid_size)
    xs, ys = np.meshgrid(axis, axis)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


class ReprojectionError(BaseMetric):
    """Mean reprojection distance over interior grid points warped by predicted vs ground-truth homographies."""

    def __init__(self, ref_corners=None, grid_size=5):
        super().__init__()
        self.ref_corners = ref_corners if ref_corners is not None else DEFAULT_REF_CORNERS
        self.grid = make_grid(grid_size)

    def __call__(self, preds, targets):
        if is_invalid_corners(preds) or is_invalid_corners(targets):
            return float("nan")
        h_pred = estimate_homography(self.ref_corners, preds)
        h_gt = estimate_homography(self.ref_corners, targets)
        proj_pred = reproject_points(self.grid, h_pred)
        proj_gt = reproject_points(self.grid, h_gt)
        return float(np.linalg.norm(proj_pred - proj_gt, axis=1).mean())
