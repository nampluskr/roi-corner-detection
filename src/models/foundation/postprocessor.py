# src/models/foundation/postprocessor.py: convert raw foundation-adapter logits into standard corners

import torch

from src.models.base.base_postprocessor import BasePostprocessor


class FoundationPostprocessor(BasePostprocessor):
    """Applies sigmoid to (N, 8) logits and reshapes to (N, 4, 2) corners."""

    def __call__(self, raw_output):
        corners = torch.sigmoid(raw_output)
        return corners.reshape(raw_output.shape[0], 4, 2)
