# src/losses/mse_loss.py: mean squared error regression loss

import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class MSELoss(BaseLoss):
    """Mean squared error loss between predicted and target tensors."""

    def forward(self, raw_output, target):
        return F.mse_loss(raw_output, target)
