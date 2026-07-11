# src/models/det/postprocessor.py: decode grid detection output into standard corners

import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import is_invalid_corners


class DetPostprocessor(BasePostprocessor):
    """Selects the top-1 cell per class from (N, A, 9) output and decodes box centers into (N, 4, 2) corners."""

    def __init__(self, score_min=0.0):
        self.score_min = score_min

    def __call__(self, raw_output):
        n, area, _ = raw_output.shape
        size = int(round(area ** 0.5))

        obj = torch.sigmoid(raw_output[..., 0])
        offset = torch.sigmoid(raw_output[..., 1:3])
        cls_prob = torch.softmax(raw_output[..., 5:9], dim=-1)
        score = obj.unsqueeze(-1) * cls_prob

        best_cell = score.argmax(dim=1)
        best_score = score.max(dim=1).values
        batch = torch.arange(n, device=raw_output.device)

        corners = torch.zeros(n, 4, 2, device=raw_output.device)
        for k in range(4):
            a = best_cell[:, k]
            col = (a % size).to(raw_output.dtype)
            row = (a // size).to(raw_output.dtype)
            corners[:, k, 0] = (col + offset[batch, a, 0]) / size
            corners[:, k, 1] = (row + offset[batch, a, 1]) / size

        corners = corners.cpu().numpy()
        best_score = best_score.cpu().numpy()
        results = [self._validate(corners[i], best_score[i]) for i in range(n)]
        return torch.from_numpy(np.stack(results))

    def _validate(self, pts, scores):
        if np.any(scores < self.score_min) or is_invalid_corners(pts):
            return np.full((4, 2), np.nan, dtype=np.float32)
        return pts.astype(np.float32)
