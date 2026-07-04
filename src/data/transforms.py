# src/data/transforms.py: joint (image, corners) transforms

import random
import math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Compose:
    """Applies a sequence of (image, corners) transforms in order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, corners):
        for t in self.transforms:
            image, corners = t(image, corners)
        return image, corners


# --- Geometric transforms: apply to image and corners simultaneously ---
# corners layout: (4, 2) normalized [0, 1], order TL, TR, BR, BL

class Resize:
    def __init__(self, size):
        self.size = size  # int or (H, W)

    def __call__(self, image, corners):
        return F.resize(image, self.size), corners


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, corners):
        if random.random() >= self.p:
            return image, corners
        image = F.hflip(image)
        tl, tr, br, bl = corners
        c = np.stack([
            [1.0 - tr[0], tr[1]],
            [1.0 - tl[0], tl[1]],
            [1.0 - bl[0], bl[1]],
            [1.0 - br[0], br[1]],
        ]).astype(np.float32)
        return image, c


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, corners):
        if random.random() >= self.p:
            return image, corners
        image = F.vflip(image)
        tl, tr, br, bl = corners
        c = np.stack([
            [bl[0], 1.0 - bl[1]],
            [br[0], 1.0 - br[1]],
            [tr[0], 1.0 - tr[1]],
            [tl[0], 1.0 - tl[1]],
        ]).astype(np.float32)
        return image, c


class RandomRotation:
    def __init__(self, degrees=5):
        self.degrees = degrees

    def __call__(self, image, corners):
        angle = random.uniform(-self.degrees, self.degrees)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        rotated = np.empty_like(corners)
        for i, (x, y) in enumerate(corners):
            dx, dy = x - 0.5, y - 0.5
            rotated[i, 0] = cos_a * dx - sin_a * dy + 0.5
            rotated[i, 1] = sin_a * dx + cos_a * dy + 0.5

        if rotated.min() < 0.0 or rotated.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        return image, rotated.astype(np.float32)


class RandomPerspective:
    def __init__(self, distortion_scale=0.1, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, image, corners):
        if random.random() >= self.p:
            return image, corners

        width, height = image.size
        half_w, half_h = self.distortion_scale * width / 2, self.distortion_scale * height / 2
        src_pts = [[0, 0], [width, 0], [width, height], [0, height]]
        dst_pts = [
            [random.uniform(0, half_w), random.uniform(0, half_h)],
            [random.uniform(width - half_w, width), random.uniform(0, half_h)],
            [random.uniform(width - half_w, width), random.uniform(height - half_h, height)],
            [random.uniform(0, half_w), random.uniform(height - half_h, height)],
        ]

        jittered = corners.copy()
        jittered[:, 0] = jittered[:, 0] * width
        jittered[:, 1] = jittered[:, 1] * height
        jittered = _apply_perspective_to_points(jittered, src_pts, dst_pts)
        jittered[:, 0] /= width
        jittered[:, 1] /= height

        if jittered.min() < 0.0 or jittered.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.perspective(image, src_pts, dst_pts, interpolation=F.InterpolationMode.BILINEAR)
        return image, jittered.astype(np.float32)


class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, image, corners):
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        new_size = (round(height * scale), round(width * scale))

        scaled = (corners - 0.5) / scale + 0.5
        if scaled.min() < 0.0 or scaled.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.resize(image, new_size)
        image = F.center_crop(image, (height, width))
        return image, scaled.astype(np.float32)


class RandomAffine:
    def __init__(self, degrees=5, translate=(0.05, 0.05), scale_range=(0.9, 1.1), shear=5):
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear

    def __call__(self, image, corners):
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx, max_dy = self.translate
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale = random.uniform(*self.scale_range)
        shear_x = random.uniform(-self.shear, self.shear)

        width, height = image.size
        center = np.array([0.5, 0.5])
        matrix = _affine_matrix(angle, (tx, ty), scale, (shear_x, 0.0))

        pts = corners - center
        transformed = pts @ matrix[:2, :2].T + matrix[:2, 2] + center

        if transformed.min() < 0.0 or transformed.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.affine(
            image, angle=angle,
            translate=[tx * width, ty * height],
            scale=scale, shear=[shear_x, 0.0],
            interpolation=F.InterpolationMode.BILINEAR,
        )
        return image, transformed.astype(np.float32)


def _affine_matrix(angle, translate, scale, shear):
    # forward affine matrix matching torchvision.transforms.functional.affine semantics
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])
    tx, ty = translate

    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    matrix = np.array([
        [a * scale, b * scale, tx],
        [c * scale, d * scale, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return matrix


def _perspective_matrix(src_pts, dst_pts):
    # solves for the 3x3 homography H mapping src_pts -> dst_pts (4-point correspondence)
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src_pts, dst_pts)):
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    h = np.append(np.linalg.solve(a, b), 1.0).reshape(3, 3)
    return h


def _apply_perspective_to_points(points, src_pts, dst_pts):
    h = _perspective_matrix(src_pts, dst_pts)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    projected = homogeneous @ h.T
    return (projected[:, :2] / projected[:, 2:3]).astype(np.float64)


# --- Image-only transforms: corners pass through unchanged ---

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self._t = T.ColorJitter(brightness=brightness, contrast=contrast,
                                saturation=saturation, hue=hue)

    def __call__(self, image, corners):
        return self._t(image), corners


class GaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self._t = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image, corners):
        return self._t(image), corners


class ToTensor:
    """Converts a PIL image and corners array to tensors."""

    def __call__(self, image, corners):
        return F.to_tensor(image), torch.tensor(corners, dtype=torch.float32)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, corners):
        return F.normalize(image, self.mean, self.std), corners


class GaussianNoise:
    """Adds Gaussian noise to an image tensor; must run after ToTensor."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image, corners):
        noise = torch.randn_like(image) * self.std
        return (image + noise).clamp(0.0, 1.0), corners
