# src/losses/wing_loss.py: Wing loss for coordinate regression

import math
import torch

from src.losses.base_loss import BaseLoss


class WingLoss(BaseLoss):
    """Wing loss: log penalty for small errors, linear penalty for large errors."""

    def __init__(self, apply_sigmoid=False, w=10.0, epsilon=2.0):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, raw_output, target):
        pred = torch.sigmoid(raw_output) if self.apply_sigmoid else raw_output
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.c,
        )
        return loss.mean()
