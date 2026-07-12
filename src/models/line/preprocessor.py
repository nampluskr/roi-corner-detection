# src/models/line/preprocessor.py: build dense M-LSD center/displacement targets from standard corners

import numpy as np
import torch

from src.models.base.base_preprocessor import BasePreprocessor
from src.models.line.model import MLSD_INPUT_SIZE

MAP_STRIDE = 2
STAMP_SPACING = 2.0
SUBSEG_HALF_FRAC = 1.0 / 16.0
GAUSSIAN_RADIUS = 2


def _gaussian_patch(radius):
    """Return a (2r+1, 2r+1) Gaussian peak normalized to 1 at the center."""
    span = np.arange(-radius, radius + 1, dtype=np.float32)
    xs, ys = np.meshgrid(span, span)
    sigma = radius / 2.0 if radius > 0 else 1.0
    return np.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


class LinePreprocessor(BasePreprocessor):
    """Converts (N, 4, 2) corners into dense center heat, sub-segment displacement, and mask targets."""

    def __init__(self):
        self.map_size = MLSD_INPUT_SIZE // MAP_STRIDE
        self.patch = _gaussian_patch(GAUSSIAN_RADIUS)

    def __call__(self, corners):
        pts = corners.detach().cpu().numpy()
        batch = pts.shape[0]
        size = self.map_size
        center = np.zeros((batch, 1, size, size), dtype=np.float32)
        disp = np.zeros((batch, 4, size, size), dtype=np.float32)
        mask = np.zeros((batch, 1, size, size), dtype=np.float32)
        for n in range(batch):
            self._draw_targets(pts[n] * size, center[n, 0], disp[n], mask[n, 0])
        targets = {
            "center": torch.from_numpy(center),
            "disp": torch.from_numpy(disp),
            "mask": torch.from_numpy(mask),
        }
        return {k: v.to(corners.device) for k, v in targets.items()}

    def _draw_targets(self, quad, center, disp, mask):
        for k in range(4):
            a = quad[k]
            b = quad[(k + 1) % 4]
            side = b - a
            length = float(np.hypot(side[0], side[1]))
            if length < 1e-6:
                continue
            direction = side / length
            half = SUBSEG_HALF_FRAC * length
            num_stamps = max(int(length / STAMP_SPACING), 1)
            for i in range(num_stamps):
                s = (i + 0.5) * length / num_stamps
                c = a + direction * s
                p1 = a + direction * max(s - half, 0.0)
                p2 = a + direction * min(s + half, length)
                self._stamp(center, disp, mask, c, p1, p2)

    def _stamp(self, center, disp, mask, c, p1, p2):
        size = self.map_size
        radius = GAUSSIAN_RADIUS
        cx = int(round(c[0]))
        cy = int(round(c[1]))
        if cx < 0 or cx >= size or cy < 0 or cy >= size:
            return
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        px0, py0 = x0 - (cx - radius), y0 - (cy - radius)
        patch = self.patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
        center[y0:y1, x0:x1] = np.maximum(center[y0:y1, x0:x1], patch)
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        disp[0, y0:y1, x0:x1] = p1[0] - xs
        disp[1, y0:y1, x0:x1] = p1[1] - ys
        disp[2, y0:y1, x0:x1] = p2[0] - xs
        disp[3, y0:y1, x0:x1] = p2[1] - ys
        mask[y0:y1, x0:x1] = 1.0
