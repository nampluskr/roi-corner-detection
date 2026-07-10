# src/losses/bce_loss.py: binary cross-entropy loss over segmentation mask logits

import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class BCELoss(BaseLoss):
    """Binary cross-entropy with logits between predicted mask logits and target mask."""

    def forward(self, raw_output, target):
        return F.binary_cross_entropy_with_logits(raw_output, target)
