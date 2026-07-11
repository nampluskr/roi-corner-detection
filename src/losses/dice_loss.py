# src/losses/dice_loss.py: soft dice loss over segmentation mask logits

import torch

from src.losses.base_loss import BaseLoss


class DiceLoss(BaseLoss):
    """Soft dice loss between sigmoid mask probabilities and target mask."""

    def __init__(self, eps=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.eps = eps

    def forward(self, raw_output, target):
        prob = torch.sigmoid(raw_output)
        intersection = (prob * target).sum()
        union = prob.sum() + target.sum()
        return 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
