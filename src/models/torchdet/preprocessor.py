# src/models/torchdet/preprocessor.py: convert standard corners into torchvision detection targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor

BOX_SIZE = 1.0 / 16


class TorchdetPreprocessor(BasePreprocessor):
    """Converts (N, 4, 2) corners into per-image {boxes, labels} target dicts for a torchvision detector."""

    def __call__(self, corners, image_size, box_size=BOX_SIZE):
        device = corners.device
        half = box_size * image_size / 2
        labels = torch.arange(1, 5, dtype=torch.long, device=device)

        targets = []
        for pts in corners:
            xy = pts * image_size
            x = xy[:, 0]
            y = xy[:, 1]
            half_t = torch.full_like(x, half)
            hw = torch.minimum(half_t, torch.minimum(x, image_size - x)).clamp(min=1.0)
            hh = torch.minimum(half_t, torch.minimum(y, image_size - y)).clamp(min=1.0)
            boxes = torch.stack([x - hw, y - hh, x + hw, y + hh], dim=1)
            targets.append({"boxes": boxes, "labels": labels})
        return targets
