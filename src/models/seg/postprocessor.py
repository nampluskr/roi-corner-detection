# src/models/seg/postprocessor.py: convert a quad mask into standard corners via contour extraction

import cv2
import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import order_corners, is_invalid_corners, mask_to_corners

EPSILON_FRACTIONS = [0.02, 0.01, 0.03, 0.05, 0.08, 0.1]


class SegPostprocessor(BasePostprocessor):
    """Thresholds (N, 1, H, W) mask logits and returns standard (N, 4, 2) corners via findContours."""

    def __call__(self, raw_output):
        prob = torch.sigmoid(raw_output)
        masks = (prob[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
        corners = [self._extract(mask) for mask in masks]
        return torch.from_numpy(np.stack(corners))

    def _extract(self, mask):
        height, width = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.full((4, 2), np.nan, dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        quad = self._approx_quad(contour)
        pts = quad / np.array([width, height], dtype=np.float32)
        pts = order_corners(pts)
        if is_invalid_corners(pts):
            fallback = order_corners(mask_to_corners(mask))
            if is_invalid_corners(fallback):
                return np.full((4, 2), np.nan, dtype=np.float32)
            return fallback
        return pts

    def _approx_quad(self, contour):
        peri = cv2.arcLength(contour, True)
        for frac in EPSILON_FRACTIONS:
            approx = cv2.approxPolyDP(contour, frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect).astype(np.float32)
