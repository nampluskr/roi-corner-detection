# src/models/torchdet/postprocessor.py: decode torchvision detections into standard corners

import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import is_invalid_corners


class TorchdetPostprocessor(BasePostprocessor):
    """Selects the top-scoring box per corner class from torchvision detections and returns (N, 4, 2) corners."""

    def __init__(self, score_min=0.0):
        self.score_min = score_min

    def __call__(self, raw_output, image_size):
        results = [self._decode(det, image_size) for det in raw_output]
        return torch.from_numpy(np.stack(results))

    def _decode(self, det, image_size):
        boxes = det["boxes"].detach().cpu().numpy()
        labels = det["labels"].detach().cpu().numpy()
        scores = det["scores"].detach().cpu().numpy()

        corners = np.full((4, 2), np.nan, dtype=np.float32)
        for c in range(1, 5):
            mask = labels == c
            if not mask.any():
                return np.full((4, 2), np.nan, dtype=np.float32)
            idx = np.argmax(np.where(mask, scores, -1.0))
            if scores[idx] < self.score_min:
                return np.full((4, 2), np.nan, dtype=np.float32)
            x1, y1, x2, y2 = boxes[idx]
            corners[c - 1, 0] = (x1 + x2) / 2.0 / image_size
            corners[c - 1, 1] = (y1 + y2) / 2.0 / image_size

        if is_invalid_corners(corners):
            return np.full((4, 2), np.nan, dtype=np.float32)
        return corners
