# src/models/heatmap/preprocessor.py: convert standard corners into per-corner gaussian heatmap targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor
from src.models.heatmap.model import HEATMAP_SIZE

SIGMA = 1.5


class HeatmapPreprocessor(BasePreprocessor):
    """Renders (N, 4, 2) normalized corners as (N, 4, S, S) gaussian heatmaps with unit peaks."""

    def __call__(self, corners):
        size = HEATMAP_SIZE
        cx = corners[..., 0] * (size - 1)
        cy = corners[..., 1] * (size - 1)
        axis = torch.arange(size, dtype=corners.dtype, device=corners.device)
        dist_x = (axis.reshape(1, 1, size) - cx.unsqueeze(-1)) ** 2
        dist_y = (axis.reshape(1, 1, size) - cy.unsqueeze(-1)) ** 2
        dist = dist_y.unsqueeze(-1) + dist_x.unsqueeze(-2)
        return torch.exp(-dist / (2.0 * SIGMA * SIGMA))
