# src/metrics/pck.py: percentage of correct keypoints (sample-level success check)

from src.metrics.base_metric import BaseMetric
from src.metrics.max_cd import MaxCD


class PCK(BaseMetric):
    """Returns whether MaxCD between predicted and ground-truth corners is within tau."""

    def __init__(self):
        self._max_cd = MaxCD()

    def __call__(self, pred_corners, gt_corners, tau):
        return self._max_cd(pred_corners, gt_corners) <= tau
