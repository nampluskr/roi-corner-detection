# src/losses/cross_entropy_loss.py: softmax cross-entropy loss over class logits

import torch.nn.functional as F

from src.losses.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Softmax cross-entropy between class logits and integer class targets."""

    def forward(self, raw_output, target):
        return F.cross_entropy(raw_output, target)
