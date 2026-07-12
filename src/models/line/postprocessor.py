# src/models/line/postprocessor.py: decode M-LSD tp-map into corners via line grouping and intersection

import cv2
import numpy as np
import torch

from src.models.base.base_postprocessor import BasePostprocessor
from src.utils.geometry import order_corners, is_invalid_corners

CENTER_CHANNEL = 0
DISP_START = 8
CENTER_THRESHOLD = 0.2
NMS_KERNEL = 3
TOPK = 200
STRICT_TOPK = 50
MIN_SEG_LEN_FRAC = 0.01
KMEANS_ITER = 20


def _decode_segments(center_map, disp_map, topk=TOPK):
    """Return (M, 4) segments [x1, y1, x2, y2] in map pixels from center heat and start/end displacement."""
    height, width = center_map.shape
    pooled = cv2.dilate(center_map, np.ones((NMS_KERNEL, NMS_KERNEL), np.float32))
    peaks = (center_map >= pooled) & (center_map > CENTER_THRESHOLD)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    scores = center_map[ys, xs]
    order = np.argsort(-scores)[:topk]
    ys, xs = ys[order], xs[order]

    start = disp_map[0:2, ys, xs].T
    end = disp_map[2:4, ys, xs].T
    centers = np.stack([xs, ys], axis=1).astype(np.float32)
    p1 = centers + start
    p2 = centers + end
    segments = np.concatenate([p1, p2], axis=1).astype(np.float32)

    min_len = MIN_SEG_LEN_FRAC * min(height, width)
    lengths = np.linalg.norm(p2 - p1, axis=1)
    return segments[lengths >= min_len]


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


def _direction_groups(segments):
    """Split segments into two orientation groups via k-means on double-angle unit vectors."""
    deltas = segments[:, 2:4] - segments[:, 0:2]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    feats = np.stack([np.cos(2 * angles), np.sin(2 * angles)], axis=1).astype(np.float32)
    if len(feats) < 2:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_ITER, 1e-3)
    _, labels, _ = cv2.kmeans(feats, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        return None
    return labels


def _split_opposite(segments):
    """Split an orientation group into two opposite sides by the sign of the normal projection."""
    mids = 0.5 * (segments[:, 0:2] + segments[:, 2:4])
    deltas = segments[:, 2:4] - segments[:, 0:2]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    double = np.stack([np.cos(2 * angles), np.sin(2 * angles)], axis=0).mean(axis=1)
    common_angle = 0.5 * np.arctan2(double[1], double[0])
    direction = np.array([np.cos(common_angle), np.sin(common_angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    proj = mids @ normal
    threshold = proj.mean()
    side_a = segments[proj >= threshold]
    side_b = segments[proj < threshold]
    if len(side_a) == 0 or len(side_b) == 0:
        return None
    return side_a, side_b


def _segments_to_points(segments):
    """Flatten (K, 4) segments into (2K, 2) endpoint coordinates."""
    return segments.reshape(-1, 2)


def _corners_from_segments(segments, height, width):
    """Group segments into 4 sides, fit lines, and return ordered normalized corners or None."""
    if len(segments) < 4:
        return None

    labels = _direction_groups(segments)
    if labels is None:
        return None

    group_lines = []
    for g in (0, 1):
        group = segments[labels == g]
        if len(group) < 2:
            return None
        split = _split_opposite(group)
        if split is None:
            return None
        group_lines.append([_fit_line(_segments_to_points(side)) for side in split])

    corners = _cross_group_corners(group_lines[0], group_lines[1])
    if corners is None:
        return None

    pts = corners / np.array([width, height], dtype=np.float32)
    pts = order_corners(pts)
    if is_invalid_corners(pts):
        return None
    return pts


def _cross_group_corners(group_a, group_b):
    """Intersect each line of group_a with each line of group_b, returning 4 corners or None."""
    corners = []
    for line_a in group_a:
        for line_b in group_b:
            point = _intersect(line_a, line_b)
            if point is None:
                return None
            corners.append(point)
    return np.array(corners, dtype=np.float32)


class LinePostprocessor(BasePostprocessor):
    """Decodes (N, 16, h, w) tp-maps into (N, 4, 2) corners via segment grouping and intersection."""

    def __call__(self, raw_output):
        raw = raw_output.detach().cpu().numpy().astype(np.float32)
        center = 1.0 / (1.0 + np.exp(-raw[:, CENTER_CHANNEL]))
        disp = raw[:, DISP_START:DISP_START + 4]
        results = [self._extract(center[i], disp[i]) for i in range(raw.shape[0])]
        return torch.from_numpy(np.stack(results).astype(np.float32))

    def _extract(self, center_map, disp_map):
        height, width = center_map.shape
        for topk in (STRICT_TOPK, TOPK):
            segments = _decode_segments(center_map, disp_map, topk=topk)
            corners = _corners_from_segments(segments, height, width)
            if corners is not None:
                return corners
        return np.full((4, 2), np.nan, dtype=np.float32)
