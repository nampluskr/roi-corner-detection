# src/models/hybrid/postprocessor.py: convert a quad mask into corners via Canny + Hough + cornerSubPix

import cv2
import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import order_corners, is_invalid_corners, mask_to_corners

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 20
HOUGH_MIN_LEN_FRAC = 0.15
HOUGH_MAX_GAP = 10
SUBPIX_WIN = 5
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
EPSILON_FRACTIONS = [0.02, 0.01, 0.03, 0.05, 0.08, 0.1]


def _fit_line(points):
    """Fit a total-least-squares line to (K, 2) points and return (a, b, c) with a*x + b*y + c = 0."""
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return vy, -vx, vx * y0 - vy * x0


def _intersect(line1, line2):
    """Intersect two lines given as (a, b, c); return (x, y) or None if near parallel."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return x, y


def _group_sides(segments, cx, cy):
    """Split Hough segments into top/bottom/left/right point sets by angle and mask centroid."""
    sides = {"top": [], "bottom": [], "left": [], "right": []}
    for x1, y1, x2, y2 in segments:
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1))
        mid_x, mid_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if angle < np.pi / 4:
            key = "top" if mid_y < cy else "bottom"
        else:
            key = "left" if mid_x < cx else "right"
        sides[key].extend([[x1, y1], [x2, y2]])
    return sides


class HybridPostprocessor(BasePostprocessor):
    """Thresholds (N, 1, H, W) mask logits and returns (N, 4, 2) corners via Canny/Hough/cornerSubPix."""

    def __call__(self, raw_output):
        prob = torch.sigmoid(raw_output)[:, 0].cpu().numpy().astype(np.float32)
        masks = (prob > 0.5).astype(np.uint8)
        corners = [self._extract(mask, prob_map) for mask, prob_map in zip(masks, prob)]
        return torch.from_numpy(np.stack(corners).astype(np.float32))

    def _extract(self, mask, prob_map):
        height, width = mask.shape
        pts = self._corners_from_lines(mask, prob_map)
        if pts is not None:
            pts = order_corners(pts / np.array([width, height], dtype=np.float32))
            if not is_invalid_corners(pts):
                return pts
        return self._extract_contour(mask)

    def _extract_contour(self, mask):
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

    def _corners_from_lines(self, mask, prob_map):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())

        edges = cv2.Canny(mask * 255, CANNY_LOW, CANNY_HIGH)
        min_len = int(min(mask.shape) * HOUGH_MIN_LEN_FRAC)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                                minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP)
        if lines is None:
            return None

        sides = _group_sides(lines.reshape(-1, 4), cx, cy)
        if any(len(sides[key]) < 2 for key in sides):
            return None

        fits = {key: _fit_line(np.array(pts, dtype=np.float32)) for key, pts in sides.items()}
        pairs = [("top", "left"), ("top", "right"), ("bottom", "right"), ("bottom", "left")]
        corners = []
        for side_a, side_b in pairs:
            point = _intersect(fits[side_a], fits[side_b])
            if point is None:
                return None
            corners.append(point)

        pts = np.array(corners, dtype=np.float32)
        return self._refine(pts, prob_map)

    def _refine(self, pts, prob_map):
        height, width = prob_map.shape
        margin = SUBPIX_WIN + 1
        if (pts[:, 0] < margin).any() or (pts[:, 0] > width - margin).any():
            return pts
        if (pts[:, 1] < margin).any() or (pts[:, 1] > height - margin).any():
            return pts
        refined = pts.reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(prob_map, refined, (SUBPIX_WIN, SUBPIX_WIN), (-1, -1), SUBPIX_CRITERIA)
        return refined.reshape(-1, 2)
