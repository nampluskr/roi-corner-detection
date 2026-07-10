# src/models/heatmap/postprocessor.py: convert per-corner heatmaps into standard corners via soft-argmax

import torch
import torch.nn.functional as F

from src.models.base.base_postprocessor import BasePostprocessor
from src.models.heatmap.model import HEATMAP_SIZE

BETA = 20.0


class HeatmapPostprocessor(BasePostprocessor):
    """Applies spatial softmax to (N, 4, S, S) heatmaps and returns expected (N, 4, 2) corners."""

    def __call__(self, raw_output):
        n, k, height, width = raw_output.shape
        prob = F.softmax(BETA * raw_output.reshape(n, k, height * width), dim=-1)
        prob = prob.reshape(n, k, height, width)
        cols = torch.arange(width, dtype=raw_output.dtype, device=raw_output.device) / (HEATMAP_SIZE - 1)
        rows = torch.arange(height, dtype=raw_output.dtype, device=raw_output.device) / (HEATMAP_SIZE - 1)
        x = (prob.sum(dim=-2) * cols).sum(dim=-1)
        y = (prob.sum(dim=-1) * rows).sum(dim=-1)
        corners = torch.stack([x, y], dim=-1)
        return corners.clamp(0.0, 1.0)
