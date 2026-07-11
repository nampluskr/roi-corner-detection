# src/losses/smooth_l1_loss.py: SmoothL1 (Huber) regression loss

import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class SmoothL1Loss(BaseLoss):
    """SmoothL1 (Huber) loss: quadratic for small errors, linear for large errors."""

    def __init__(self, beta=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta

    def forward(self, raw_output, target):
        return F.smooth_l1_loss(raw_output, target, beta=self.beta)
