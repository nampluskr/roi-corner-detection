# src/data/transforms.py: joint (image, corners) transforms

import random
import math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    """Applies a sequence of (image, corners) transforms in order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, corners=None):
        if corners is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, corners = t(image, corners)
        return image, corners


# --- Geometric transforms: apply to image and corners simultaneously ---
# corners layout: (4, 2) normalized [0, 1], order TL, TR, BR, BL

class Resize:
    """Resizes the image to a fixed size; corners are unaffected (already normalized)."""

    def __init__(self, size):
        self.size = size  # int or (H, W)

    def __call__(self, image, corners=None):
        image = F.resize(image, self.size)
        return image if corners is None else (image, corners)


class RandomHorizontalFlip:
    """Randomly flips the image horizontally and reorders corners to match."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, corners=None):
        if random.random() >= self.p:
            return image if corners is None else (image, corners)
        image = F.hflip(image)
        if corners is None:
            return image
        tl, tr, br, bl = corners
        c = np.stack([
            [1.0 - tr[0], tr[1]],
            [1.0 - tl[0], tl[1]],
            [1.0 - bl[0], bl[1]],
            [1.0 - br[0], br[1]],
        ]).astype(np.float32)
        return image, c


class RandomVerticalFlip:
    """Randomly flips the image vertically and reorders corners to match."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, corners=None):
        if random.random() >= self.p:
            return image if corners is None else (image, corners)
        image = F.vflip(image)
        if corners is None:
            return image
        tl, tr, br, bl = corners
        c = np.stack([
            [bl[0], 1.0 - bl[1]],
            [br[0], 1.0 - br[1]],
            [tr[0], 1.0 - tr[1]],
            [tl[0], 1.0 - tl[1]],
        ]).astype(np.float32)
        return image, c


class RandomRotation:
    """Randomly rotates the image in pixel space, skipping if any corner exits [0, 1]."""

    def __init__(self, degrees=5):
        self.degrees = degrees

    def __call__(self, image, corners=None):
        angle = random.uniform(-self.degrees, self.degrees)

        if corners is None:
            return F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)

        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # rotate in pixel space so non-square aspect ratios are not distorted
        # F.rotate turns clockwise in image (y-down) pixel coordinates for a positive angle
        width, height = image.size
        pts = corners * np.array([width, height]) - np.array([width / 2.0, height / 2.0])
        rotated = np.empty_like(pts)
        rotated[:, 0] = cos_a * pts[:, 0] + sin_a * pts[:, 1]
        rotated[:, 1] = -sin_a * pts[:, 0] + cos_a * pts[:, 1]
        rotated = (rotated + np.array([width / 2.0, height / 2.0])) / np.array([width, height])

        if rotated.min() < 0.0 or rotated.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        return image, rotated.astype(np.float32)


class RandomPerspective:
    """Randomly warps the image with a perspective transform, skipping if any corner exits [0, 1]."""

    def __init__(self, distortion_scale=0.1, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, image, corners=None):
        if random.random() >= self.p:
            return image if corners is None else (image, corners)

        width, height = image.size
        half_w, half_h = self.distortion_scale * width / 2, self.distortion_scale * height / 2
        src_pts = [[0, 0], [width, 0], [width, height], [0, height]]
        dst_pts = [
            [random.uniform(0, half_w), random.uniform(0, half_h)],
            [random.uniform(width - half_w, width), random.uniform(0, half_h)],
            [random.uniform(width - half_w, width), random.uniform(height - half_h, height)],
            [random.uniform(0, half_w), random.uniform(height - half_h, height)],
        ]

        if corners is None:
            return F.perspective(image, src_pts, dst_pts, interpolation=F.InterpolationMode.BILINEAR)

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
    """Randomly scales the image with a center crop back to the original size."""

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, image, corners=None):
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        new_size = (round(height * scale), round(width * scale))

        if corners is None:
            image = F.resize(image, new_size)
            return F.center_crop(image, (height, width))

        scaled = (corners - 0.5) * scale + 0.5
        if scaled.min() < 0.0 or scaled.max() > 1.0:
            return image, corners  # skip if any corner falls outside [0, 1]

        image = F.resize(image, new_size)
        image = F.center_crop(image, (height, width))
        return image, scaled.astype(np.float32)


class RandomAffine:
    """Randomly applies rotation, translation, scale, and shear in pixel space."""

    def __init__(self, degrees=5, translate=(0.05, 0.05), scale_range=(0.9, 1.1), shear=5):
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear

    def __call__(self, image, corners=None):
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx, max_dy = self.translate
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale = random.uniform(*self.scale_range)
        shear_x = random.uniform(-self.shear, self.shear)

        width, height = image.size

        if corners is None:
            return F.affine(
                image, angle=angle,
                translate=[tx * width, ty * height],
                scale=scale, shear=[shear_x, 0.0],
                interpolation=F.InterpolationMode.BILINEAR,
            )

        # apply the affine transform in pixel space so non-square aspect ratios are not distorted
        size = np.array([width, height])
        center = size / 2.0
        matrix = _affine_matrix(angle, (tx * width, ty * height), scale, (shear_x, 0.0))

        pts = corners * size - center
        transformed = pts @ matrix[:2, :2].T + matrix[:2, 2] + center
        transformed = transformed / size

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
    # solves for the 3x3 homography H mapping src_pts $\to$ dst_pts (4-point correspondence)
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
    """Randomly perturbs brightness, contrast, saturation, and hue; corners pass through unchanged."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self._t = T.ColorJitter(brightness=brightness, contrast=contrast,
                                saturation=saturation, hue=hue)

    def __call__(self, image, corners=None):
        image = self._t(image)
        return image if corners is None else (image, corners)


class GaussianBlur:
    """Applies Gaussian blur to the image; corners pass through unchanged."""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self._t = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image, corners=None):
        image = self._t(image)
        return image if corners is None else (image, corners)


class ToTensor:
    """Converts a PIL image and corners array to tensors."""

    def __call__(self, image, corners=None):
        image = F.to_tensor(image)
        return image if corners is None else (image, torch.tensor(corners, dtype=torch.float32))


class Normalize:
    """Normalizes an image tensor with the given per-channel mean and std."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, corners=None):
        image = F.normalize(image, self.mean, self.std)
        return image if corners is None else (image, corners)


class Denormalize:
    """Reverse Normalize: restore original pixel scale with x * std + mean."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image, corners=None):
        image = image * self.std + self.mean
        return image if corners is None else (image, corners)


class ToNumpy:
    """Converts a (3, H, W) image tensor to a clamped (H, W, 3) numpy array, and corners to numpy."""

    def __call__(self, image, corners=None):
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
        return image if corners is None else (image, corners.numpy())


class GaussianNoise:
    """Adds Gaussian noise to an image tensor; must run after ToTensor."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image, corners=None):
        noise = torch.randn_like(image) * self.std
        image = (image + noise).clamp(0.0, 1.0)
        return image if corners is None else (image, corners)
