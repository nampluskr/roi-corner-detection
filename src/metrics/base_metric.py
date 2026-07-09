# src/metrics/base_metric.py: base class for stateful sample-level metrics

import numpy as np


class BaseMetric:
    """Base class for a stateful (reset/update/compute) sample-level metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            if np.isnan(pred).any():
                continue
            value = self(pred, target)
            if isinstance(value, float) and np.isnan(value):
                continue
            self.total += value
            self.count += 1

    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

    def __call__(self, preds, targets):
        raise NotImplementedError
