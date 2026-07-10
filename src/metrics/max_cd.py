# src/metrics/max_cd.py: max corner distance between predicted and ground-truth corners

import numpy as np

from src.metrics.base_metric import BaseMetric


class MaxCD(BaseMetric):
    """Computes the maximum Euclidean distance over 4 corresponding corner pairs."""

    def __call__(self, preds, targets):
        pred = np.array(preds, dtype=np.float64).reshape(4, 2)
        target = np.array(targets, dtype=np.float64).reshape(4, 2)
        return float(np.linalg.norm(pred - target, axis=1).max())
