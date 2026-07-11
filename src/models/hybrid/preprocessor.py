# src/models/hybrid/preprocessor.py: convert standard corners into a hybrid mask target

import torch

from src.models.base.base_preprocessor import BasePreprocessor


class HybridPreprocessor(BasePreprocessor):
    """Rasterizes normalized corners as filled quad masks for hybrid mask training."""

    def __call__(self, corners, size=None):
        if size is None:
            raise ValueError("HybridPreprocessor requires the target mask size")

        px = corners[..., 0] * (size - 1)
        py = corners[..., 1] * (size - 1)
        vx = px.unsqueeze(-1).unsqueeze(-1)
        vy = py.unsqueeze(-1).unsqueeze(-1)
        ex = torch.roll(px, -1, dims=-1).unsqueeze(-1).unsqueeze(-1) - vx
        ey = torch.roll(py, -1, dims=-1).unsqueeze(-1).unsqueeze(-1) - vy

        axis = torch.arange(size, dtype=corners.dtype, device=corners.device)
        grid_x = axis.reshape(1, 1, 1, size)
        grid_y = axis.reshape(1, 1, size, 1)
        cross = ex * (grid_y - vy) - ey * (grid_x - vx)

        inside_pos = (cross >= 0.0).all(dim=1)
        inside_neg = (cross <= 0.0).all(dim=1)
        mask = (inside_pos | inside_neg).to(corners.dtype)
        return mask.unsqueeze(1)
