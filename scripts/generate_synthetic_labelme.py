# scripts/generate_synthetic_labelme.py: generate synthetic OLED fringe images and LabelMe polygons.

import argparse
import json
import math
import os
import re

import cv2
import numpy as np
from PIL import Image


PREVIEW_LEFT_RIGHT = {1, 2, 5, 6, 9, 11, 14, 15, 16, 18}
PREVIEW_VISIBLE = {1, 4, 7, 10, 13, 16, 19}
PREVIEW_PARTIAL = {2, 5, 8, 11, 14, 17, 20}
PREVIEW_ROTATIONS = [
    -9.0, -6.0, -3.0, 0.0, 3.0,
    6.0, 9.0, 7.0, 4.0, 1.0,
    -2.0, -5.0, -8.0, 8.0, 5.0,
    2.0, -1.0, -4.0, -7.0, 10.0,
]
PREVIEW_HORIZONTAL_SHIFTS = [
    -0.08, -0.04, 0.00, 0.04, 0.08,
    0.00, 0.08, -0.04, 0.04, -0.08,
    0.04, -0.08, 0.08, -0.04, 0.00,
    -0.04, 0.04, -0.08, 0.08, 0.00,
]
PREVIEW_TOP_SCALES = [
    0.86, 0.90, 0.94, 0.88, 0.92,
    0.95, 0.87, 0.91, 0.96, 0.89,
    0.93, 0.86, 0.90, 0.94, 0.88,
    0.92, 0.95, 0.87, 0.91, 0.96,
]
PREVIEW_03_LEFT_RIGHT = {1, 2, 6, 10, 14, 15, 19}
PREVIEW_03_TOP_BOTTOM = {3, 4, 8, 11, 13, 17, 18}
PREVIEW_03_ALL_SIDES = {5, 7, 9, 12, 16, 20}
PREVIEW_03_FREQUENCIES = [
    8, 16, 10, 26, 34,
    9, 14, 22, 12, 36,
    18, 8, 28, 35, 11,
    15, 24, 10, 30, 36,
]
PREVIEW_03_SEVERE = {4, 9, 14, 19}
PREVIEW_03_MODERATE = {2, 3, 7, 8, 12, 13, 17, 18}
PREVIEW_03_WEAK_GRADIENT = {1, 8, 13, 20}
PREVIEW_03_CONTRASTS = [
    "soft", "medium", "clear", "medium", "soft",
    "medium", "soft", "medium", "clear", "soft",
    "clear", "medium", "soft", "medium", "soft",
    "medium", "clear", "soft", "medium", "clear",
]
PREVIEW_03_HOLDER_COUNTS = {
    1: (1, 0),
    2: (2, 0),
    3: (0, 1),
    4: (0, 2),
    5: (1, 2),
    6: (3, 0),
    7: (2, 1),
    8: (0, 3),
    9: (2, 3),
    10: (2, 0),
    11: (0, 2),
    12: (3, 2),
    13: (0, 1),
    14: (1, 0),
    15: (3, 0),
    16: (1, 3),
    17: (0, 3),
    18: (0, 2),
    19: (2, 0),
    20: (3, 1),
}
BACKGROUND_NAMES = [
    "dark_stage",
    "medium_stage",
    "bright_stage",
    "gradient_stage",
    "textured_stage",
]


def parse_args(default_condition="preview_02", forced_geometry_profile=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=default_condition)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-dir", default=None)
    if forced_geometry_profile is None:
        parser.add_argument(
            "--geometry-profile",
            choices=["preview_01", "preview_02", "preview_03"],
            default="preview_02",
        )
    else:
        parser.set_defaults(geometry_profile=forced_geometry_profile)
    return parser.parse_args()


def validate_args(args):
    """Validate generator arguments before creating output."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.condition) is None:
        raise ValueError("condition must contain only letters, digits, '_' or '-'")
    if args.count < 1 or args.count > 9999:
        raise ValueError("count must be between 1 and 9999")
    if args.width < 320 or args.height < 320:
        raise ValueError("width and height must be at least 320")


def list_backgrounds(root):
    """Return a stable list of optional texture image paths."""
    if root is None or not os.path.isdir(root):
        return []
    paths = []
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(os.path.join(current, name))
    return paths


def preview_spec(index, count, geometry_profile="preview_02"):
    """Return the controlled condition assignment for one sample."""
    if count == 20:
        group = (index - 1) // 5
        shape = "rectangle" if group < 2 else "square"
        direction = "horizontal" if group % 2 == 0 else "vertical"
        background = BACKGROUND_NAMES[(index - 1) % 5]
        fixture = "left_right" if index in PREVIEW_LEFT_RIGHT else "top_bottom"
        hole_position = "top_center" if index % 2 == 1 else "upper_left"
        if index in PREVIEW_VISIBLE:
            hole_visibility = "visible"
        elif index in PREVIEW_PARTIAL:
            hole_visibility = "partial"
        else:
            hole_visibility = "hidden"
    else:
        combinations = [
            ("rectangle", "horizontal"),
            ("rectangle", "vertical"),
            ("square", "horizontal"),
            ("square", "vertical"),
        ]
        shape, direction = combinations[(index - 1) % len(combinations)]
        background = BACKGROUND_NAMES[(index - 1) % len(BACKGROUND_NAMES)]
        fixture = "left_right" if index % 2 == 1 else "top_bottom"
        hole_position = "top_center" if (index // 2) % 2 == 0 else "upper_left"
        hole_visibility = ["visible", "partial", "hidden"][(index - 1) % 3]
    spec = {
        "shape": shape,
        "direction": direction,
        "background": background,
        "fixture": fixture,
        "hole_position": hole_position,
        "hole_visibility": hole_visibility,
    }
    if geometry_profile != "preview_03":
        return spec

    profile_index = (index - 1) % 20
    if count == 20:
        if index in PREVIEW_03_LEFT_RIGHT:
            spec["fixture"] = "left_right"
        elif index in PREVIEW_03_TOP_BOTTOM:
            spec["fixture"] = "top_bottom"
        else:
            spec["fixture"] = "all_sides"
        lr_count, tb_count = PREVIEW_03_HOLDER_COUNTS[index]
        if index in PREVIEW_03_SEVERE:
            deformation = "severe"
        elif index in PREVIEW_03_MODERATE:
            deformation = "moderate"
        else:
            deformation = "mild"
        frequency = PREVIEW_03_FREQUENCIES[profile_index]
        gradient = "weak" if index in PREVIEW_03_WEAK_GRADIENT else "full"
    else:
        layouts = ["left_right", "top_bottom", "all_sides"]
        spec["fixture"] = layouts[profile_index % len(layouts)]
        lr_count = 1 + profile_index % 3 if spec["fixture"] != "top_bottom" else 0
        tb_count = 1 + (profile_index + 1) % 3 if spec["fixture"] != "left_right" else 0
        deformation = ["mild", "moderate", "severe"][profile_index % 3]
        frequency = PREVIEW_03_FREQUENCIES[profile_index]
        gradient = "weak" if profile_index % 5 == 0 else "full"
    spec.update({
        "frequency": frequency,
        "deformation": deformation,
        "contrast": PREVIEW_03_CONTRASTS[profile_index],
        "illumination_gradient": gradient,
        "holder_lr_count": lr_count,
        "holder_tb_count": tb_count,
        "holder_taper_class": ["rectangle", "mild_trapezoid", "trapezoid"][
            profile_index % 3
        ],
    })
    return spec


def resize_crop_gray(image, width, height, rng):
    """Resize and crop an image to the target size."""
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, int(round(source_width * scale)))
    resized_height = max(height, int(round(source_height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x0 = int(rng.integers(0, resized_width - width + 1))
    y0 = int(rng.integers(0, resized_height - height + 1))
    return resized[y0:y0 + height, x0:x0 + width]


def multiscale_noise(width, height, rng):
    """Create smooth low-contrast surface variation."""
    result = np.zeros((height, width), dtype=np.float32)
    for grid, weight in [(8, 0.55), (24, 0.30), (64, 0.15)]:
        small_width = max(2, int(math.ceil(width / grid)))
        small_height = max(2, int(math.ceil(height / grid)))
        small = rng.normal(0.0, 1.0, (small_height, small_width)).astype(np.float32)
        layer = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        result += weight * layer
    result -= float(result.mean())
    std = float(result.std())
    if std > 1e-6:
        result /= std
    return result


def add_stage_hardware(stage, rng):
    """Add subtle industrial stage seams, rails and fasteners."""
    height, width = stage.shape
    canvas = np.clip(stage * 255.0, 0, 255).astype(np.uint8)
    mean = int(np.mean(canvas))
    line_dark = int(np.clip(mean - rng.uniform(12, 35), 5, 220))
    line_light = int(np.clip(mean + rng.uniform(8, 28), 15, 245))

    layout = int(rng.integers(0, 4))
    if layout in {0, 2}:
        y = int(rng.uniform(0.05, 0.12) * height)
        cv2.line(canvas, (0, y), (width - 1, y), line_dark, 5, cv2.LINE_AA)
        cv2.line(canvas, (0, y + 7), (width - 1, y + 7), line_light, 2, cv2.LINE_AA)
    if layout in {1, 2}:
        x = int(rng.uniform(0.04, 0.10) * width)
        cv2.line(canvas, (x, 0), (x, height - 1), line_dark, 5, cv2.LINE_AA)
        cv2.line(canvas, (x + 7, 0), (x + 7, height - 1), line_light, 2, cv2.LINE_AA)
    if layout == 3:
        y = int(rng.uniform(0.88, 0.95) * height)
        cv2.line(canvas, (0, y), (width - 1, y - int(0.02 * height)), line_dark, 4, cv2.LINE_AA)

    screw_count = int(rng.integers(2, 7))
    for _ in range(screw_count):
        side = int(rng.integers(0, 4))
        if side == 0:
            x = int(rng.uniform(0.02, 0.98) * width)
            y = int(rng.uniform(0.025, 0.10) * height)
        elif side == 1:
            x = int(rng.uniform(0.02, 0.98) * width)
            y = int(rng.uniform(0.90, 0.975) * height)
        elif side == 2:
            x = int(rng.uniform(0.02, 0.09) * width)
            y = int(rng.uniform(0.15, 0.85) * height)
        else:
            x = int(rng.uniform(0.91, 0.98) * width)
            y = int(rng.uniform(0.15, 0.85) * height)
        radius = int(rng.uniform(5, 11) * height / 1080.0)
        radius = max(radius, 3)
        cv2.circle(canvas, (x + 2, y + 3), radius + 2, line_dark, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, line_light, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x - 2, y - 2), max(1, radius // 3), min(250, line_light + 25), -1, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def create_stage(width, height, background_name, rng, texture_paths):
    """Create a continuous stage surface under varied illumination."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)
    if background_name == "dark_stage":
        target_mean = float(rng.uniform(0.07, 0.17))
        gradient_strength = float(rng.uniform(0.03, 0.10))
    elif background_name == "medium_stage":
        target_mean = float(rng.uniform(0.28, 0.48))
        gradient_strength = float(rng.uniform(0.04, 0.16))
    elif background_name == "bright_stage":
        target_mean = float(rng.uniform(0.62, 0.82))
        gradient_strength = float(rng.uniform(0.05, 0.18))
    elif background_name == "gradient_stage":
        target_mean = float(rng.uniform(0.30, 0.62))
        gradient_strength = float(rng.uniform(0.22, 0.46))
    else:
        target_mean = float(rng.uniform(0.16, 0.72))
        gradient_strength = float(rng.uniform(0.06, 0.20))

    angle = float(rng.uniform(0, 2 * math.pi))
    linear = (xn - 0.5) * math.cos(angle) + (yn - 0.5) * math.sin(angle)
    center_x = float(rng.uniform(0.15, 0.85))
    center_y = float(rng.uniform(0.10, 0.90))
    radial = np.sqrt((xn - center_x) ** 2 + (yn - center_y) ** 2)
    radial -= float(radial.mean())
    illumination = 0.65 * linear - 0.35 * radial
    stage = target_mean + gradient_strength * illumination

    smooth = multiscale_noise(width, height, rng)
    stage += float(rng.uniform(0.006, 0.025)) * smooth

    if background_name == "textured_stage" and texture_paths:
        path = texture_paths[int(rng.integers(0, len(texture_paths)))]
        texture = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if texture is not None:
            texture = resize_crop_gray(texture, width, height, rng).astype(np.float32) / 255.0
            low = cv2.GaussianBlur(texture, (0, 0), sigmaX=max(width, height) / 80.0)
            high = texture - low
            high_std = float(high.std())
            if high_std > 1e-6:
                high /= high_std
            stage += float(rng.uniform(0.018, 0.045)) * high

    stage += rng.normal(0.0, rng.uniform(0.002, 0.008), stage.shape).astype(np.float32)
    stage += target_mean - float(stage.mean())
    stage = np.clip(stage, 0.015, 0.96)
    return add_stage_hardware(stage, rng)


def polygon_area(points):
    """Return the absolute area of a polygon."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def sample_quad_preview_01(width, height, shape, rng):
    """Sample the large near-top-view quad used for preview_01."""
    frame_area = width * height
    for _ in range(500):
        if shape == "rectangle":
            aspect = float(rng.uniform(1.50, 1.90))
            target_area = float(rng.uniform(0.55, 0.67)) * frame_area
            panel_height = math.sqrt(target_area / aspect)
            panel_width = panel_height * aspect
            max_width = width * 0.92
            max_height = height * 0.88
            scale = min(1.0, max_width / panel_width, max_height / panel_height)
            panel_width *= scale
            panel_height *= scale
            rotation_limit = 3.0
            jitter_fraction = 0.018
        else:
            aspect = float(rng.uniform(0.98, 1.02))
            target_area = float(rng.uniform(0.505, 0.525)) * frame_area
            panel_height = math.sqrt(target_area / aspect)
            panel_width = panel_height * aspect
            scale = min(1.0, width * 0.94 / panel_width, height * 0.975 / panel_height)
            panel_width *= scale
            panel_height *= scale
            rotation_limit = 0.65
            jitter_fraction = 0.005

        center_x = width * 0.5 + float(rng.uniform(-0.025, 0.025)) * width
        center_y = height * 0.5 + float(rng.uniform(-0.012, 0.012)) * height
        corners = np.array([
            [-panel_width / 2, -panel_height / 2],
            [panel_width / 2, -panel_height / 2],
            [panel_width / 2, panel_height / 2],
            [-panel_width / 2, panel_height / 2],
        ], dtype=np.float32)
        angle = math.radians(float(rng.uniform(-rotation_limit, rotation_limit)))
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], dtype=np.float32)
        quad = corners @ rotation.T
        jitter = rng.uniform(-1.0, 1.0, (4, 2)).astype(np.float32)
        jitter[:, 0] *= panel_width * jitter_fraction
        jitter[:, 1] *= panel_height * jitter_fraction
        quad += jitter
        quad[:, 0] += center_x
        quad[:, 1] += center_y

        margin = 5.0
        if np.any(quad[:, 0] <= margin) or np.any(quad[:, 0] >= width - margin):
            continue
        if np.any(quad[:, 1] <= margin) or np.any(quad[:, 1] >= height - margin):
            continue
        contour = quad.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(contour):
            continue
        if polygon_area(quad) < 0.5 * frame_area:
            continue
        canonical_width = max(320, int(round(panel_width)))
        canonical_height = max(320, int(round(panel_height)))
        return quad, canonical_width, canonical_height
    raise RuntimeError("could not sample a valid preview_01 panel quad")


def sample_quad_preview_02(width, height, shape, index, rng):
    """Sample a smaller tilted trapezoidal reference quad."""
    frame_area = width * height
    for _ in range(500):
        if shape == "rectangle":
            aspect = float(rng.uniform(1.50, 1.90))
            target_area = float(rng.uniform(0.375, 0.460)) * frame_area
            panel_height = math.sqrt(target_area / aspect)
            panel_width = panel_height * aspect
            max_width = width * 0.78
            max_height = height * 0.76
            scale = min(1.0, max_width / panel_width, max_height / panel_height)
            panel_width *= scale
            panel_height *= scale
        else:
            aspect = float(rng.uniform(0.98, 1.02))
            target_area = float(rng.uniform(0.345, 0.375)) * frame_area
            panel_height = math.sqrt(target_area / aspect)
            panel_width = panel_height * aspect
            scale = min(1.0, width * 0.58 / panel_width, height * 0.82 / panel_height)
            panel_width *= scale
            panel_height *= scale

        profile_index = (index - 1) % len(PREVIEW_ROTATIONS)
        center_x = width * (
            0.5 + PREVIEW_HORIZONTAL_SHIFTS[profile_index] + float(rng.uniform(-0.008, 0.008))
        )
        center_y = height * (0.5 + float(rng.uniform(-0.025, 0.025)))
        top_scale = PREVIEW_TOP_SCALES[profile_index] + float(rng.uniform(-0.008, 0.008))
        top_offset = float(rng.uniform(-0.025, 0.025)) * panel_width
        corners = np.array([
            [-panel_width * top_scale / 2 + top_offset, -panel_height / 2],
            [panel_width * top_scale / 2 + top_offset, -panel_height / 2],
            [panel_width / 2, panel_height / 2],
            [-panel_width / 2, panel_height / 2],
        ], dtype=np.float32)
        rotation_degrees = PREVIEW_ROTATIONS[profile_index] + float(rng.uniform(-0.6, 0.6))
        rotation_degrees = float(np.clip(rotation_degrees, -10.0, 10.0))
        angle = math.radians(rotation_degrees)
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], dtype=np.float32)
        quad = corners @ rotation.T
        jitter = rng.uniform(-1.0, 1.0, (4, 2)).astype(np.float32)
        jitter[:, 0] *= panel_width * 0.004
        jitter[:, 1] *= panel_height * 0.004
        quad += jitter
        quad[:, 0] += center_x
        quad[:, 1] += center_y

        margin = 5.0
        if np.any(quad[:, 0] <= margin) or np.any(quad[:, 0] >= width - margin):
            continue
        if np.any(quad[:, 1] <= margin) or np.any(quad[:, 1] >= height - margin):
            continue
        contour = quad.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(contour):
            continue
        area_ratio = polygon_area(quad) / frame_area
        if area_ratio < 0.30 or area_ratio > 0.46:
            continue
        top_length = float(np.linalg.norm(quad[1] - quad[0]))
        bottom_length = float(np.linalg.norm(quad[2] - quad[3]))
        trapezoid_ratio = top_length / max(bottom_length, 1e-6)
        if trapezoid_ratio < 0.84 or trapezoid_ratio > 0.98:
            continue
        canonical_width = max(320, int(round(panel_width)))
        canonical_height = max(320, int(round(panel_height)))
        return quad, canonical_width, canonical_height
    raise RuntimeError("could not sample a valid preview_02 panel quad")


def sample_quad(width, height, shape, index, rng, geometry_profile):
    """Dispatch quad sampling to the requested preview geometry profile."""
    if geometry_profile == "preview_01":
        return sample_quad_preview_01(width, height, shape, rng)
    if geometry_profile in {"preview_02", "preview_03"}:
        return sample_quad_preview_02(width, height, shape, index, rng)
    raise ValueError("unknown geometry profile: " + geometry_profile)


def rounded_mask(width, height, radius):
    """Create a rounded rectangle mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(np.clip(radius, 1, min(width, height) // 2))
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    centers = [
        (radius, radius),
        (width - radius - 1, radius),
        (width - radius - 1, height - radius - 1),
        (radius, height - radius - 1),
    ]
    for center in centers:
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def create_panel(canonical_width, canonical_height, spec, index, rng):
    """Create a canonical rounded OLED with distorted monochrome fringe."""
    width = canonical_width
    height = canonical_height
    short_side = min(width, height)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)

    radius_fraction = float(rng.uniform(0.03, 0.08))
    radius = int(round(short_side * radius_fraction))
    outer_mask = rounded_mask(width, height, radius)
    bezel = max(4, int(round(short_side * rng.uniform(0.008, 0.018))))
    inner_width = width - 2 * bezel
    inner_height = height - 2 * bezel
    inner_radius = max(2, radius - bezel)
    active_mask = np.zeros_like(outer_mask)
    active_mask[bezel:bezel + inner_height, bezel:bezel + inner_width] = rounded_mask(
        inner_width, inner_height, inner_radius
    )

    if spec["hole_position"] == "top_center":
        hole_x = float(rng.uniform(0.47, 0.53) * width)
        hole_y = float(rng.uniform(0.055, 0.095) * height)
    else:
        hole_x = float(rng.uniform(0.13, 0.22) * width)
        hole_y = float(rng.uniform(0.055, 0.10) * height)

    if spec["hole_visibility"] == "hidden":
        frequency = 12.0
        diameter_fraction = 0.03
    else:
        frequency = float(12 + ((index * 7 + int(rng.integers(0, 4))) % 17))
        diameter_fraction = float(rng.uniform(0.032, 0.058))
    hole_radius = max(4, int(round(short_side * diameter_fraction * 0.5)))

    if spec["direction"] == "horizontal":
        axis = yn
        orthogonal = xn
        hole_axis = hole_y / max(1, height - 1)
        hole_orthogonal = hole_x / max(1, width - 1)
    else:
        axis = xn
        orthogonal = yn
        hole_axis = hole_x / max(1, width - 1)
        hole_orthogonal = hole_y / max(1, height - 1)

    bend_cycles = float(rng.uniform(0.7, 2.4))
    bend_phase = float(rng.uniform(0, 2 * math.pi))
    bend_amplitude = float(rng.uniform(0.18, 0.72))
    bend = bend_amplitude * np.sin(2 * math.pi * bend_cycles * orthogonal + bend_phase)
    bend_at_hole = bend_amplitude * math.sin(
        2 * math.pi * bend_cycles * hole_orthogonal + bend_phase
    )

    bump_x = float(rng.uniform(0.15, 0.85))
    bump_y = float(rng.uniform(0.15, 0.85))
    bump_sigma = float(rng.uniform(0.09, 0.24))
    bump_amplitude = float(rng.uniform(-0.65, 0.65))
    bump = bump_amplitude * np.exp(
        -((xn - bump_x) ** 2 + (yn - bump_y) ** 2) / (2 * bump_sigma ** 2)
    )
    hole_xn = hole_x / max(1, width - 1)
    hole_yn = hole_y / max(1, height - 1)
    bump_at_hole = bump_amplitude * math.exp(
        -((hole_xn - bump_x) ** 2 + (hole_yn - bump_y) ** 2) / (2 * bump_sigma ** 2)
    )

    base_field = 2 * math.pi * frequency * axis + bend + bump
    target_phase = {
        "visible": 0.0,
        "partial": math.pi / 2,
        "hidden": math.pi,
    }[spec["hole_visibility"]]
    phase = target_phase - (
        2 * math.pi * frequency * hole_axis + bend_at_hole + bump_at_hole
    )

    bias = float(rng.uniform(0.38, 0.58))
    amplitude = float(rng.uniform(0.34, min(0.49, bias - 0.025, 0.975 - bias)))
    fringe = bias + amplitude * np.cos(base_field + phase)

    light_angle = float(rng.uniform(0, 2 * math.pi))
    panel_gradient = (xn - 0.5) * math.cos(light_angle) + (yn - 0.5) * math.sin(light_angle)
    fringe *= 1.0 + float(rng.uniform(-0.18, 0.18)) * panel_gradient
    distance = np.sqrt(((xn - 0.5) / 0.72) ** 2 + ((yn - 0.5) / 0.72) ** 2)
    fringe *= 1.0 - float(rng.uniform(0.0, 0.20)) * np.clip(distance, 0, 1)

    if index % 3 == 0 or float(rng.random()) < 0.35:
        glare_x = float(rng.uniform(0.15, 0.85))
        glare_y = float(rng.uniform(0.15, 0.85))
        glare_sx = float(rng.uniform(0.05, 0.20))
        glare_sy = float(rng.uniform(0.08, 0.30))
        glare = np.exp(
            -((xn - glare_x) ** 2 / (2 * glare_sx ** 2) +
              (yn - glare_y) ** 2 / (2 * glare_sy ** 2))
        )
        fringe += float(rng.uniform(0.04, 0.20)) * glare

    panel = np.full((height, width), float(rng.uniform(0.025, 0.09)), dtype=np.float32)
    panel = panel * (1.0 - active_mask) + np.clip(fringe, 0.015, 0.985) * active_mask

    cv2.circle(
        panel,
        (int(round(hole_x)), int(round(hole_y))),
        hole_radius,
        float(rng.uniform(0.005, 0.025)),
        -1,
        cv2.LINE_AA,
    )
    blur_sigma = float(rng.uniform(0.35, 1.15))
    panel = cv2.GaussianBlur(panel, (0, 0), sigmaX=blur_sigma)
    edge = cv2.Canny((outer_mask * 255).astype(np.uint8), 80, 160).astype(np.float32) / 255.0
    panel = np.clip(panel + edge * float(rng.uniform(0.02, 0.10)), 0, 1)
    panel *= outer_mask
    return panel, outer_mask


def stage_mean_under_quad(stage, quad):
    """Measure the stage mean below the projected OLED region."""
    mask = np.zeros(stage.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
    values = stage[mask > 127]
    if values.size == 0:
        return float(np.mean(stage))
    return float(np.median(values))


def smooth_random_field(width, height, rng):
    """Create a normalized smooth two-dimensional phase field."""
    grid_width = int(rng.integers(5, 9))
    grid_height = int(rng.integers(5, 9))
    grid = rng.normal(0.0, 1.0, (grid_height, grid_width)).astype(np.float32)
    field = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    sigma = max(width, height) / float(rng.uniform(20.0, 35.0))
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)
    field -= float(field.mean())
    maximum = float(np.max(np.abs(field)))
    if maximum > 1e-6:
        field /= maximum
    return field


def create_panel_preview_03(canonical_width, canonical_height, spec, rng, stage_mean):
    """Create a low-contrast OLED with multi-scale distorted fringe."""
    width = canonical_width
    height = canonical_height
    short_side = min(width, height)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)

    radius = int(round(short_side * rng.uniform(0.03, 0.08)))
    outer_mask = rounded_mask(width, height, radius)
    bezel = max(4, int(round(short_side * rng.uniform(0.010, 0.022))))
    inner_width = width - 2 * bezel
    inner_height = height - 2 * bezel
    inner_radius = max(2, radius - bezel)
    active_mask = np.zeros_like(outer_mask)
    active_mask[bezel:bezel + inner_height, bezel:bezel + inner_width] = rounded_mask(
        inner_width,
        inner_height,
        inner_radius,
    )

    if spec["hole_position"] == "top_center":
        hole_x = float(rng.uniform(0.47, 0.53) * width)
        hole_y = float(rng.uniform(0.055, 0.095) * height)
    else:
        hole_x = float(rng.uniform(0.13, 0.22) * width)
        hole_y = float(rng.uniform(0.055, 0.10) * height)

    frequency = float(spec["frequency"])
    if spec["hole_visibility"] == "hidden":
        diameter_fraction = 0.03
    else:
        diameter_fraction = float(rng.uniform(0.032, 0.058))
    hole_radius = max(4, int(round(short_side * diameter_fraction * 0.5)))

    if spec["direction"] == "horizontal":
        axis = yn
        orthogonal = xn
    else:
        axis = xn
        orthogonal = yn

    severity_ranges = {
        "mild": {
            "bow": (0.03, 0.08),
            "random": (0.02, 0.06),
            "bump": (0.02, 0.07),
            "chirp": (0.03, 0.10),
            "bumps": 2,
            "limit": 0.15,
        },
        "moderate": {
            "bow": (0.08, 0.16),
            "random": (0.06, 0.13),
            "bump": (0.07, 0.16),
            "chirp": (0.10, 0.22),
            "bumps": 3,
            "limit": 0.35,
        },
        "severe": {
            "bow": (0.16, 0.28),
            "random": (0.12, 0.22),
            "bump": (0.14, 0.28),
            "chirp": (0.20, 0.35),
            "bumps": 4,
            "limit": 0.70,
        },
    }
    settings = severity_ranges[spec["deformation"]]
    bow_amplitude = float(rng.uniform(*settings["bow"]))
    bow_sign = -1.0 if rng.random() < 0.5 else 1.0
    bow = bow_sign * bow_amplitude * 4.0 * (orthogonal - 0.5) ** 2
    bow -= float(bow.mean())

    random_amplitude = float(rng.uniform(*settings["random"]))
    displacement = bow + random_amplitude * smooth_random_field(width, height, rng)
    for _ in range(settings["bumps"]):
        center_x = float(rng.uniform(0.12, 0.88))
        center_y = float(rng.uniform(0.12, 0.88))
        sigma_x = float(rng.uniform(0.05, 0.22))
        sigma_y = float(rng.uniform(0.05, 0.22))
        bump_amplitude = float(rng.uniform(*settings["bump"]))
        if rng.random() < 0.5:
            bump_amplitude *= -1.0
        displacement += bump_amplitude * np.exp(
            -(
                (xn - center_x) ** 2 / (2 * sigma_x ** 2)
                + (yn - center_y) ** 2 / (2 * sigma_y ** 2)
            )
        )
    displacement -= float(displacement.mean())
    displacement_max = float(np.max(np.abs(displacement)))
    if displacement_max > settings["limit"]:
        displacement *= settings["limit"] / displacement_max

    chirp = float(rng.uniform(*settings["chirp"]))
    if rng.random() < 0.5:
        chirp *= -1.0
    cycles = frequency * axis + frequency * chirp * (axis - 0.5) ** 2 + displacement

    hole_ix = int(np.clip(round(hole_x), 0, width - 1))
    hole_iy = int(np.clip(round(hole_y), 0, height - 1))
    target_phase = {
        "visible": 0.0,
        "partial": math.pi / 2,
        "hidden": math.pi,
    }[spec["hole_visibility"]]
    phase = target_phase - 2 * math.pi * float(cycles[hole_iy, hole_ix])

    amplitude_ranges = {
        "soft": (0.10, 0.18),
        "medium": (0.18, 0.28),
        "clear": (0.28, 0.36),
    }
    bias = float(np.clip(stage_mean + rng.uniform(-0.08, 0.12), 0.20, 0.78))
    amplitude = float(rng.uniform(*amplitude_ranges[spec["contrast"]]))
    amplitude = min(amplitude, bias - 0.025, 0.975 - bias)

    light_angle = float(rng.uniform(0, 2 * math.pi))
    linear = (xn - 0.5) * math.cos(light_angle) + (yn - 0.5) * math.sin(light_angle)
    light_x = float(rng.uniform(0.15, 0.85))
    light_y = float(rng.uniform(0.15, 0.85))
    light_sx = float(rng.uniform(0.20, 0.55))
    light_sy = float(rng.uniform(0.20, 0.55))
    radial = np.exp(
        -(
            (xn - light_x) ** 2 / (2 * light_sx ** 2)
            + (yn - light_y) ** 2 / (2 * light_sy ** 2)
        )
    )
    illumination = 0.65 * linear + 0.35 * radial
    illumination -= float(illumination.min())
    illumination_range = float(illumination.max())
    if illumination_range > 1e-6:
        illumination /= illumination_range
    illumination -= 0.5
    if spec["illumination_gradient"] == "weak":
        gradient_peak = float(rng.uniform(0.02, 0.06))
    else:
        gradient_peak = float(rng.uniform(0.08, 0.35))

    modulation = 1.0 + 0.12 * smooth_random_field(width, height, rng)
    fringe = (
        bias
        + gradient_peak * illumination
        + amplitude * modulation * np.cos(2 * math.pi * cycles + phase)
    )
    fringe = np.clip(fringe, 0.015, 0.985)

    bezel_value = float(np.clip(stage_mean + rng.uniform(-0.04, 0.04), 0.02, 0.94))
    panel = np.full((height, width), bezel_value, dtype=np.float32)
    panel = panel * (1.0 - active_mask) + fringe * active_mask

    if spec["hole_visibility"] == "hidden":
        hole_value = float(np.clip(fringe[hole_iy, hole_ix] - 0.015, 0.01, 0.20))
    else:
        hole_value = float(rng.uniform(0.008, 0.035))
    cv2.circle(
        panel,
        (hole_ix, hole_iy),
        hole_radius,
        hole_value,
        -1,
        cv2.LINE_AA,
    )
    panel_blur = float(rng.uniform(0.50, 1.60))
    panel = cv2.GaussianBlur(panel, (0, 0), sigmaX=panel_blur)
    panel *= outer_mask

    spec["actual_frequency"] = frequency
    spec["gradient_peak_to_peak"] = gradient_peak
    spec["edge_feather_sigma"] = float(rng.uniform(1.5, 4.0))
    return panel, outer_mask


def warp_panel(stage, panel, mask, quad):
    """Warp a canonical OLED onto the stage with a contact shadow."""
    height, width = stage.shape
    canonical_height, canonical_width = panel.shape
    source = np.array([
        [0, 0],
        [canonical_width - 1, 0],
        [canonical_width - 1, canonical_height - 1],
        [0, canonical_height - 1],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped_panel = cv2.warpPerspective(
        panel,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shadow_transform = np.float32([[1, 0, 8], [0, 1, 10]])
    shadow = cv2.warpAffine(warped_mask, shadow_transform, (width, height))
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=10.0)
    composed = stage * (1.0 - 0.28 * shadow)
    alpha = np.clip(warped_mask, 0, 1)
    return composed * (1.0 - alpha) + warped_panel * alpha


def warp_panel_preview_03(stage, panel, mask, quad, edge_sigma, rng):
    """Warp and feather a preview_03 OLED into the stage."""
    height, width = stage.shape
    canonical_height, canonical_width = panel.shape
    source = np.array([
        [0, 0],
        [canonical_width - 1, 0],
        [canonical_width - 1, canonical_height - 1],
        [0, canonical_height - 1],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped_panel = cv2.warpPerspective(
        panel,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    numerator = cv2.GaussianBlur(warped_panel, (0, 0), sigmaX=edge_sigma)
    alpha = cv2.GaussianBlur(warped_mask, (0, 0), sigmaX=edge_sigma)
    normalized_panel = np.divide(
        numerator,
        np.maximum(alpha, 1e-5),
        out=np.zeros_like(numerator),
        where=alpha > 1e-5,
    )

    shadow_transform = np.float32([
        [1, 0, float(rng.uniform(4, 10))],
        [0, 1, float(rng.uniform(5, 12))],
    ])
    shadow = cv2.warpAffine(warped_mask, shadow_transform, (width, height))
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=float(rng.uniform(12, 24)))
    shadow_strength = float(rng.uniform(0.10, 0.22))
    composed = stage * (1.0 - shadow_strength * shadow)
    alpha = np.clip(alpha, 0, 1)
    composed = composed * (1.0 - alpha) + normalized_panel * alpha

    binary = (warped_mask >= 0.5).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    inner = binary - cv2.erode(binary, kernel)
    outer = cv2.dilate(binary, kernel) - binary
    inner_values = composed[inner > 0]
    outer_values = composed[outer > 0]
    if inner_values.size and outer_values.size:
        boundary_jump = abs(float(np.median(inner_values)) - float(np.median(outer_values)))
    else:
        boundary_jump = 1.0
    return composed, warped_mask, boundary_jump


def fixture_edges(fixture):
    """Return opposing quad edge indices for a fixture orientation."""
    if fixture == "left_right":
        return [(3, 0), (1, 2)]
    return [(0, 1), (2, 3)]


def add_fixture_blocks(image, quad, fixture, rng, index):
    """Render two dark fixture blocks touching opposite OLED edges."""
    height, width = image.shape
    center = np.mean(quad, axis=0)
    canvas = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    short_side = min(
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[2] - quad[1])),
    )
    for edge_number, (start_index, end_index) in enumerate(fixture_edges(fixture)):
        start = quad[start_index].astype(np.float64)
        end = quad[end_index].astype(np.float64)
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        unit = edge / max(edge_length, 1e-6)
        midpoint = (start + end) * 0.5
        outward = midpoint - center
        outward /= max(float(np.linalg.norm(outward)), 1e-6)

        position = float(rng.uniform(0.40, 0.60))
        block_length = edge_length * float(rng.uniform(0.09, 0.18))
        contact_center = start + position * edge
        p0 = contact_center - unit * block_length * 0.5
        p1 = contact_center + unit * block_length * 0.5
        depth = short_side * float(rng.uniform(0.045, 0.095))
        chamfer = min(block_length * 0.12, depth * 0.25)
        outer0 = p0 + outward * depth + unit * chamfer
        outer1 = p1 + outward * depth - unit * chamfer
        polygon = np.array([p0, p1, outer1, outer0], dtype=np.int32)

        shadow_polygon = polygon + np.array([5, 7], dtype=np.int32)
        shadow_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(shadow_mask, shadow_polygon, 180, cv2.LINE_AA)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=6.0)
        shadow_alpha = shadow_mask.astype(np.float32) / 255.0 * 0.30
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - shadow_alpha), 0, 255).astype(np.uint8)

        block_value = int(rng.uniform(24, 70))
        cv2.fillConvexPoly(canvas, polygon, block_value, cv2.LINE_AA)
        cv2.polylines(canvas, [polygon], True, min(120, block_value + 38), 2, cv2.LINE_AA)
        cv2.line(
            canvas,
            tuple(np.round(p0 + outward * 3).astype(int)),
            tuple(np.round(p1 + outward * 3).astype(int)),
            min(150, block_value + 55),
            2,
            cv2.LINE_AA,
        )

        if (index + edge_number) % 3 == 0:
            screw_center = contact_center + outward * depth * 0.58
            radius = max(3, int(round(depth * 0.11)))
            center_point = tuple(np.round(screw_center).astype(int))
            cv2.circle(canvas, center_point, radius + 2, max(8, block_value - 15), -1, cv2.LINE_AA)
            cv2.circle(canvas, center_point, radius, min(185, block_value + 65), -1, cv2.LINE_AA)
            slot_start = screw_center - unit * radius * 0.65
            slot_end = screw_center + unit * radius * 0.65
            cv2.line(
                canvas,
                tuple(np.round(slot_start).astype(int)),
                tuple(np.round(slot_end).astype(int)),
                max(10, block_value - 20),
                1,
                cv2.LINE_AA,
            )
    return canvas.astype(np.float32) / 255.0


def fill_gradient_polygon(canvas, polygon, contact_center, outward, depth, base, delta):
    """Fill a polygon with a directional brightness gradient."""
    height, width = canvas.shape
    x0 = max(0, int(np.min(polygon[:, 0])) - 2)
    y0 = max(0, int(np.min(polygon[:, 1])) - 2)
    x1 = min(width, int(np.max(polygon[:, 0])) + 3)
    y1 = min(height, int(np.max(polygon[:, 1])) + 3)
    if x0 >= x1 or y0 >= y1:
        return
    local_polygon = polygon.copy().astype(np.int32)
    local_polygon[:, 0] -= x0
    local_polygon[:, 1] -= y0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_polygon, 255, cv2.LINE_AA)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    projection = (
        (xx - contact_center[0]) * outward[0]
        + (yy - contact_center[1]) * outward[1]
    ) / max(depth, 1e-6)
    values = np.clip(base + delta * (projection - 0.5), 0.06, 0.50)
    alpha = mask.astype(np.float32) / 255.0
    crop = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = crop * (1.0 - alpha) + values * alpha


def preview_03_holder_edges(spec):
    """Return preview_03 holder edges and block counts."""
    edges = []
    lr_count = spec["holder_lr_count"]
    tb_count = spec["holder_tb_count"]
    if lr_count:
        edges.extend([(3, 0, lr_count), (1, 2, lr_count)])
    if tb_count:
        edges.extend([(0, 1, tb_count), (2, 3, tb_count)])
    return edges


def add_fixture_blocks_preview_03(image, quad, spec, rng, index):
    """Render varied multi-block preview_03 OLED holders."""
    height, width = image.shape
    center = np.mean(quad, axis=0).astype(np.float64)
    canvas = image.copy().astype(np.float32)
    holder_mask = np.zeros((height, width), dtype=np.uint8)
    short_side = min(
        float(np.linalg.norm(quad[1] - quad[0])),
        float(np.linalg.norm(quad[2] - quad[1])),
    )
    taper_ranges = {
        "rectangle": (0.95, 1.00),
        "mild_trapezoid": (0.78, 0.94),
        "trapezoid": (0.60, 0.77),
    }
    base_taper = float(rng.uniform(*taper_ranges[spec["holder_taper_class"]]))
    base_value = float(rng.uniform(0.06, 0.50))
    gradient_delta = float(rng.uniform(0.04, 0.16))
    rendered_count = 0

    for edge_number, (start_index, end_index, count) in enumerate(
        preview_03_holder_edges(spec)
    ):
        start = quad[start_index].astype(np.float64)
        end = quad[end_index].astype(np.float64)
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        unit = edge / max(edge_length, 1e-6)
        midpoint = (start + end) * 0.5
        outward = midpoint - center
        outward /= max(float(np.linalg.norm(outward)), 1e-6)

        positions = (np.arange(count, dtype=np.float64) + 1.0) / (count + 1.0)
        positions += rng.uniform(-0.03, 0.03, count)
        positions = np.clip(positions, 0.15, 0.85)
        for block_index, position in enumerate(positions):
            max_length = min(0.20, 0.62 / count)
            length_fraction = float(rng.uniform(0.06, max_length))
            block_length = edge_length * length_fraction
            depth = short_side * float(rng.uniform(0.03, 0.12))
            contact_center = start + float(position) * edge
            p0 = contact_center - unit * block_length * 0.5
            p1 = contact_center + unit * block_length * 0.5
            taper = float(np.clip(base_taper + rng.uniform(-0.035, 0.035), 0.60, 1.00))
            outer_center = contact_center + outward * depth
            outer0 = outer_center - unit * block_length * taper * 0.5
            outer1 = outer_center + unit * block_length * taper * 0.5
            polygon = np.round([p0, p1, outer1, outer0]).astype(np.int32)

            shadow_polygon = polygon + np.array([4, 6], dtype=np.int32)
            shadow_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(shadow_mask, shadow_polygon, 255, cv2.LINE_AA)
            shadow_mask = cv2.GaussianBlur(
                shadow_mask,
                (0, 0),
                sigmaX=float(rng.uniform(4, 10)),
            )
            shadow_alpha = shadow_mask.astype(np.float32) / 255.0
            canvas *= 1.0 - float(rng.uniform(0.10, 0.28)) * shadow_alpha

            block_value = float(np.clip(base_value + rng.uniform(-0.04, 0.04), 0.06, 0.50))
            fill_gradient_polygon(
                canvas,
                polygon,
                contact_center,
                outward,
                depth,
                block_value,
                gradient_delta,
            )
            block_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(block_mask, polygon, 255, cv2.LINE_AA)
            if np.count_nonzero((block_mask > 127) & (holder_mask > 127)):
                raise RuntimeError("preview_03 holder blocks overlap")
            holder_mask = np.maximum(holder_mask, block_mask)
            outline_value = float(np.clip(block_value + 0.12, 0.0, 0.78))
            contact_value = float(np.clip(block_value + 0.18, 0.0, 0.82))
            cv2.polylines(canvas, [polygon], True, outline_value, 2, cv2.LINE_AA)
            cv2.line(
                canvas,
                tuple(np.round(p0 + outward * 2).astype(int)),
                tuple(np.round(p1 + outward * 2).astype(int)),
                contact_value,
                2,
                cv2.LINE_AA,
            )

            if (index + edge_number + block_index) % 3 == 0:
                screw_center = contact_center + outward * depth * 0.58
                radius = max(2, int(round(depth * 0.09)))
                screw_point = tuple(np.round(screw_center).astype(int))
                cv2.circle(
                    canvas,
                    screw_point,
                    radius + 2,
                    max(0.02, block_value - 0.08),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    canvas,
                    screw_point,
                    radius,
                    min(0.85, block_value + 0.25),
                    -1,
                    cv2.LINE_AA,
                )
            rendered_count += 1

    panel_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(panel_mask, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
    panel_inner = cv2.erode(panel_mask, np.ones((7, 7), dtype=np.uint8))
    active_overlap = int(np.count_nonzero((holder_mask > 127) & (panel_inner > 127)))
    expected_count = 2 * spec["holder_lr_count"] + 2 * spec["holder_tb_count"]
    if rendered_count != expected_count:
        raise RuntimeError("preview_03 holder count mismatch")
    if active_overlap:
        raise RuntimeError("preview_03 holder overlaps the active ROI")
    return np.clip(canvas, 0, 1), holder_mask, rendered_count


def add_camera_response(image, rng, index):
    """Apply mild optical blur, noise and RGB response variation."""
    sigma = 0.20 + 0.10 * (index % 7)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    noise_sigma = float(rng.uniform(1.0, 6.0)) / 255.0
    image = np.clip(image + rng.normal(0.0, noise_sigma, image.shape), 0, 1)
    gains = np.array([
        float(rng.uniform(0.985, 1.005)),
        float(rng.uniform(0.990, 1.010)),
        float(rng.uniform(0.995, 1.015)),
    ], dtype=np.float32)
    rgb = np.stack([image, image, image], axis=-1) * gains.reshape(1, 1, 3)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def add_camera_response_preview_03(image, rng, index):
    """Apply a softer preview_03 optical and sensor response."""
    sigma = float(rng.uniform(0.35, 1.15))
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    noise_sigma = float(rng.uniform(1.0, 8.0)) / 255.0
    image = np.clip(image + rng.normal(0.0, noise_sigma, image.shape), 0, 1)
    gains = np.array([
        float(rng.uniform(0.985, 1.005)),
        float(rng.uniform(0.990, 1.010)),
        float(rng.uniform(0.995, 1.015)),
    ], dtype=np.float32)
    rgb = np.stack([image, image, image], axis=-1) * gains.reshape(1, 1, 3)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def labelme_payload(image_name, width, height, quad):
    """Build a LabelMe-compatible JSON payload."""
    points = []
    for point in quad:
        points.append([round(float(point[0]), 6), round(float(point[1]), 6)])
    return {
        "version": "6.2.0",
        "flags": {},
        "shapes": [
            {
                "label": "roi",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def validate_pair(image_path, json_path, width, height, geometry_profile):
    """Validate one generated PNG and LabelMe polygon."""
    with Image.open(image_path) as image:
        if image.size != (width, height) or image.mode != "RGB":
            raise RuntimeError("invalid image size or mode: " + image_path)
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload["imagePath"] != os.path.basename(image_path):
        raise RuntimeError("imagePath mismatch: " + json_path)
    if payload["imageWidth"] != width or payload["imageHeight"] != height:
        raise RuntimeError("JSON dimensions mismatch: " + json_path)
    if len(payload["shapes"]) != 1 or payload["shapes"][0]["label"] != "roi":
        raise RuntimeError("invalid LabelMe shapes: " + json_path)
    points = np.asarray(payload["shapes"][0]["points"], dtype=np.float32)
    if points.shape != (4, 2):
        raise RuntimeError("ROI must have four points: " + json_path)
    if np.any(points[:, 0] <= 0) or np.any(points[:, 0] >= width):
        raise RuntimeError("ROI x coordinate outside image: " + json_path)
    if np.any(points[:, 1] <= 0) or np.any(points[:, 1] >= height):
        raise RuntimeError("ROI y coordinate outside image: " + json_path)
    if not cv2.isContourConvex(points.reshape(-1, 1, 2)):
        raise RuntimeError("ROI is not convex: " + json_path)
    area_ratio = polygon_area(points) / (width * height)
    if geometry_profile == "preview_01":
        if area_ratio < 0.50:
            raise RuntimeError("preview_01 ROI area is below 50 percent: " + json_path)
    else:
        if area_ratio < 0.30 or area_ratio > 0.46:
            raise RuntimeError(geometry_profile + " ROI area is outside the expected range: " + json_path)
        top_length = float(np.linalg.norm(points[1] - points[0]))
        bottom_length = float(np.linalg.norm(points[2] - points[3]))
        trapezoid_ratio = top_length / max(bottom_length, 1e-6)
        if trapezoid_ratio < 0.84 or trapezoid_ratio > 0.98:
            raise RuntimeError(geometry_profile + " ROI is not the expected trapezoid: " + json_path)


def generate_sample(
    output_dir,
    width,
    height,
    seed,
    index,
    count,
    texture_paths,
    geometry_profile,
):
    """Generate and save one synthetic image and LabelMe JSON pair."""
    rng = np.random.default_rng(seed + index * 1009)
    spec = preview_spec(index, count, geometry_profile)
    stage = create_stage(width, height, spec["background"], rng, texture_paths)
    quad, canonical_width, canonical_height = sample_quad(
        width,
        height,
        spec["shape"],
        index,
        rng,
        geometry_profile,
    )
    if geometry_profile == "preview_03":
        local_stage_mean = stage_mean_under_quad(stage, quad)
        panel, panel_mask = create_panel_preview_03(
            canonical_width,
            canonical_height,
            spec,
            rng,
            local_stage_mean,
        )
        composed, warped_mask, boundary_jump = warp_panel_preview_03(
            stage,
            panel,
            panel_mask,
            quad,
            spec["edge_feather_sigma"],
            rng,
        )
        if boundary_jump > 0.18:
            raise RuntimeError(
                "preview_03 boundary jump exceeds 0.18: {:.4f}".format(boundary_jump)
            )
        composed, holder_mask, holder_count = add_fixture_blocks_preview_03(
            composed,
            quad,
            spec,
            rng,
            index,
        )
        spec["boundary_jump"] = boundary_jump
        spec["rendered_holder_count"] = holder_count
        rgb = add_camera_response_preview_03(composed, rng, index)
    else:
        panel, panel_mask = create_panel(canonical_width, canonical_height, spec, index, rng)
        composed = warp_panel(stage, panel, panel_mask, quad)
        composed = add_fixture_blocks(composed, quad, spec["fixture"], rng, index)
        rgb = add_camera_response(composed, rng, index)

    stem = "synthetic_{:04d}".format(index)
    image_name = stem + ".png"
    image_path = os.path.join(output_dir, image_name)
    json_path = os.path.join(output_dir, stem + ".json")
    Image.fromarray(rgb, mode="RGB").save(image_path, format="PNG", compress_level=5)
    payload = labelme_payload(image_name, width, height, quad)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    validate_pair(image_path, json_path, width, height, geometry_profile)
    return spec, polygon_area(quad) / (width * height)


def validate_output(output_dir, count, geometry_profile):
    """Validate the final file set and preview condition distribution."""
    names = sorted(os.listdir(output_dir))
    expected = []
    for index in range(1, count + 1):
        stem = "synthetic_{:04d}".format(index)
        expected.extend([stem + ".json", stem + ".png"])
    if names != sorted(expected):
        raise RuntimeError("output directory does not contain exactly the expected PNG/JSON pairs")
    if count == 20:
        specs = [
            preview_spec(index, count, geometry_profile)
            for index in range(1, count + 1)
        ]
        combinations = {}
        for spec in specs:
            key = (spec["shape"], spec["direction"])
            combinations[key] = combinations.get(key, 0) + 1
        if sorted(combinations.values()) != [5, 5, 5, 5]:
            raise RuntimeError("invalid preview OLED/fringe distribution")
        visibility_counts = {
            name: sum(spec["hole_visibility"] == name for spec in specs)
            for name in ["visible", "partial", "hidden"]
        }
        if visibility_counts != {"visible": 7, "partial": 7, "hidden": 6}:
            raise RuntimeError("invalid preview hole visibility distribution")
        if geometry_profile == "preview_03":
            fixture_counts = {
                name: sum(spec["fixture"] == name for spec in specs)
                for name in ["left_right", "top_bottom", "all_sides"]
            }
            deformation_counts = {
                name: sum(spec["deformation"] == name for spec in specs)
                for name in ["mild", "moderate", "severe"]
            }
            gradient_counts = {
                name: sum(spec["illumination_gradient"] == name for spec in specs)
                for name in ["full", "weak"]
            }
            if fixture_counts != {"left_right": 7, "top_bottom": 7, "all_sides": 6}:
                raise RuntimeError("invalid preview_03 holder layout distribution")
            if deformation_counts != {"mild": 8, "moderate": 8, "severe": 4}:
                raise RuntimeError("invalid preview_03 deformation distribution")
            if gradient_counts != {"full": 16, "weak": 4}:
                raise RuntimeError("invalid preview_03 illumination distribution")
            frequencies = [spec["frequency"] for spec in specs]
            if min(frequencies) != 8 or max(frequencies) != 36:
                raise RuntimeError("preview_03 frequency range must span 8 to 36 cycles")
            for spec in specs:
                for count_key in ["holder_lr_count", "holder_tb_count"]:
                    holder_count = spec[count_key]
                    if holder_count < 0 or holder_count > 3:
                        raise RuntimeError("preview_03 holder count must be between 0 and 3")
        else:
            fixture_counts = {
                name: sum(spec["fixture"] == name for spec in specs)
                for name in ["left_right", "top_bottom"]
            }
            if fixture_counts != {"left_right": 10, "top_bottom": 10}:
                raise RuntimeError("invalid preview fixture distribution")


def main(default_condition="preview_02", forced_geometry_profile=None):
    """Generate a condition folder without overwriting existing output."""
    args = parse_args(default_condition, forced_geometry_profile)
    validate_args(args)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "data", "synthetic", args.condition)
    if os.path.exists(output_dir):
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            pass
        else:
            raise FileExistsError("output directory already exists and is not empty: " + output_dir)
    else:
        os.makedirs(output_dir)

    texture_paths = list_backgrounds(args.background_dir)
    print("output=" + output_dir)
    print("geometry_profile=" + args.geometry_profile)
    print("textures={}".format(len(texture_paths)))
    for index in range(1, args.count + 1):
        spec, area_ratio = generate_sample(
            output_dir,
            args.width,
            args.height,
            args.seed,
            index,
            args.count,
            texture_paths,
            args.geometry_profile,
        )
        if args.geometry_profile == "preview_03":
            print(
                (
                    "{:04d} shape={} fringe={} freq={} deform={} contrast={} "
                    "stage={} fixture={} holders={}/{} hole={}/{} area={:.3f} edge={:.3f}"
                ).format(
                    index,
                    spec["shape"],
                    spec["direction"],
                    int(spec["actual_frequency"]),
                    spec["deformation"],
                    spec["contrast"],
                    spec["background"],
                    spec["fixture"],
                    spec["holder_lr_count"],
                    spec["holder_tb_count"],
                    spec["hole_position"],
                    spec["hole_visibility"],
                    area_ratio,
                    spec["boundary_jump"],
                )
            )
        else:
            print(
                "{:04d} shape={} fringe={} stage={} fixture={} hole={}/{} area={:.3f}".format(
                    index,
                    spec["shape"],
                    spec["direction"],
                    spec["background"],
                    spec["fixture"],
                    spec["hole_position"],
                    spec["hole_visibility"],
                    area_ratio,
                )
            )
    validate_output(output_dir, args.count, args.geometry_profile)
    print("validated_pairs={}".format(args.count))


if __name__ == "__main__":
    main()
