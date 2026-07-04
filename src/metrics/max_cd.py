# src/metrics/max_cd.py: max corner distance between predicted and ground-truth corners

import numpy as np

from src.metrics.base_metric import BaseMetric


class MaxCD(BaseMetric):
    """Computes the maximum Euclidean distance over 4 corresponding corner pairs."""

    def __call__(self, pred_corners, gt_corners):
        pred = np.array(pred_corners, dtype=np.float64).reshape(4, 2)
        gt = np.array(gt_corners, dtype=np.float64).reshape(4, 2)
        return float(np.linalg.norm(pred - gt, axis=1).max())
