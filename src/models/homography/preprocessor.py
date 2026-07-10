# src/models/homography/preprocessor.py: convert standard corners into canonical-offset targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor
from src.models.homography.model import CANONICAL_CORNERS


class HomographyPreprocessor(BasePreprocessor):
    """Subtracts canonical vertices from (N, 4, 2) corners and flattens to (N, 8) offset targets."""

    def __call__(self, corners):
        canonical = torch.tensor(CANONICAL_CORNERS, dtype=corners.dtype, device=corners.device)
        offsets = corners - canonical
        return offsets.reshape(corners.shape[0], 8)
