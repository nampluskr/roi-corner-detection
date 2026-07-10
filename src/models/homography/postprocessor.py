# src/models/homography/postprocessor.py: convert raw offsets into standard corners via canonical vertices

import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.models.homography.model import ALPHA, CANONICAL_CORNERS


class HomographyPostprocessor(BasePostprocessor):
    """Applies alpha*tanh to (N, 8) offsets, adds canonical vertices, and clamps to (N, 4, 2) corners."""

    def __call__(self, raw_output):
        canonical = torch.tensor(CANONICAL_CORNERS, dtype=raw_output.dtype, device=raw_output.device)
        offsets = ALPHA * torch.tanh(raw_output)
        corners = offsets.reshape(raw_output.shape[0], 4, 2) + canonical
        return corners.clamp(0.0, 1.0)
