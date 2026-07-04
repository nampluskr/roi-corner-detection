# src/metrics/base_metric.py: base class for sample-level corner detection metrics

class BaseMetric:
    """Base class for sample-level metrics comparing predicted and ground-truth corners."""

    def __call__(self, pred_corners, gt_corners):
        raise NotImplementedError
