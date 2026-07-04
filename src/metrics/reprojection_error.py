# src/metrics/reprojection_error.py: homography reprojection error between predicted and ground-truth corners

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.utils.homography import estimate_homography, reproject_points


class ReprojectionError(BaseMetric):
    """Computes mean reprojection distance between homographies fit from predicted and ground-truth corners."""

    def __call__(self, pred_corners, gt_corners, ref_corners):
        h_pred = estimate_homography(ref_corners, pred_corners)
        h_gt = estimate_homography(ref_corners, gt_corners)
        proj_pred = reproject_points(ref_corners, h_pred)
        proj_gt = reproject_points(ref_corners, h_gt)
        return float(np.linalg.norm(proj_pred - proj_gt, axis=1).mean())
