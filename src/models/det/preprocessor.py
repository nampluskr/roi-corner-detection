# src/models/det/preprocessor.py: convert standard corners into grid detection targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor

BOX_SIZE = 1.0 / 16


class DetPreprocessor(BasePreprocessor):
    """Assigns each of (N, 4, 2) corners to its grid cell and builds obj/box/class/mask targets."""

    def __call__(self, corners, size=None, box_size=BOX_SIZE):
        if size is None:
            raise ValueError("DetPreprocessor requires the grid size")

        n = corners.shape[0]
        area = size * size
        device = corners.device
        x = corners[..., 0]
        y = corners[..., 1]

        j = (x * size).floor().clamp(0, size - 1).long()
        i = (y * size).floor().clamp(0, size - 1).long()
        cell = i * size + j
        dx = x * size - j.to(corners.dtype)
        dy = y * size - i.to(corners.dtype)
        half = box_size / 2
        w = 2.0 * torch.minimum(torch.minimum(x, 1.0 - x), torch.full_like(x, half))
        h = 2.0 * torch.minimum(torch.minimum(y, 1.0 - y), torch.full_like(y, half))

        obj_t = torch.zeros(n, area, dtype=corners.dtype, device=device)
        box_t = torch.zeros(n, area, 4, dtype=corners.dtype, device=device)
        cls_t = torch.zeros(n, area, dtype=torch.long, device=device)
        pos_mask = torch.zeros(n, area, dtype=torch.bool, device=device)

        batch = torch.arange(n, device=device)
        for k in range(4):
            a = cell[:, k]
            obj_t[batch, a] = 1.0
            box_t[batch, a] = torch.stack([dx[:, k], dy[:, k], w[:, k], h[:, k]], dim=-1)
            cls_t[batch, a] = k
            pos_mask[batch, a] = True

        return obj_t, box_t, cls_t, pos_mask
