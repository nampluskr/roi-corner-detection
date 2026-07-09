# src/metrics/success_rate.py: fraction of predicted corners that are valid (non-NaN)

import numpy as np

from src.metrics.base_metric import BaseMetric


class SuccessRate(BaseMetric):
    """Fraction of predictions where postprocessing produced valid (non-NaN) corners."""

    def update(self, preds, targets):
        for pred in preds:
            self.count += 1
            if not np.isnan(pred).any():
                self.total += 1.0

    def __call__(self, preds, targets):
        return 0.0 if np.isnan(preds).any() else 1.0
