# src/metrics/pck.py: percentage of correct keypoints (sample-level success check)

from src.metrics.base_metric import BaseMetric
from src.metrics.max_cd import MaxCD


class PCK(BaseMetric):
    """Returns whether MaxCD between predicted and ground-truth corners is within tau."""

    def __init__(self, tau=0.02):
        super().__init__()
        self._max_cd = MaxCD()
        self.tau = tau

    def __call__(self, preds, targets):
        return self._max_cd(preds, targets) <= self.tau
