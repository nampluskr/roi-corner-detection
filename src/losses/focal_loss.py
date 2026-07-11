# src/losses/focal_loss.py: binary focal loss for class-imbalanced objectness prediction

import torch
import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class FocalLoss(BaseLoss):
    """Binary focal loss down-weighting easy negatives via a (1 - p_t)^gamma factor."""

    def __init__(self, alpha=0.25, gamma=2.0, weight=1.0):
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, raw_output, target):
        bce = F.binary_cross_entropy_with_logits(raw_output, target, reduction="none")
        prob = torch.sigmoid(raw_output)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()
