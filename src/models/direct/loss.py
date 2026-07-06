# src/models/direct/loss.py: sigmoid + Wing Loss for direct coordinate regression

import math
import torch

from src.models.base.base_loss import BaseLoss


class _WingLoss:
    """Wing loss: log-based penalty for small errors, linear penalty for large errors."""

    def __init__(self, w=10.0, epsilon=2.0):
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def __call__(self, pred, target):
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.c,
        )
        return loss.mean()


class DirectLoss(BaseLoss):
    """Applies sigmoid to raw logits then computes Wing loss against the normalized-coordinate target."""

    def __init__(self, w=10.0, epsilon=2.0):
        self.wing_loss = _WingLoss(w=w, epsilon=epsilon)

    def __call__(self, raw_output, target):
        pred = torch.sigmoid(raw_output)
        return self.wing_loss(pred, target)
