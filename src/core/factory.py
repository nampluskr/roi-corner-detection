# src/core/factory.py: factory functions for creating data pipeline objects

import numpy as np
import torch
from src.data.dataset import Dataset
from src.data.dataloader import Dataloader
from src.data.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    ColorJitter, GaussianBlur,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(split, input_size=512):
    """Return a Compose of (image, corners) transforms for the given split."""
    transforms = []
    if split == "train":
        transforms += [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=5.0),
            ColorJitter(brightness=0.2, contrast=0.2),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    transforms += [
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return Compose(transforms)


def get_dataset(split, csv_path, input_size=512):
    """Return a Dataset for the given split, built from csv_path with split-specific transform."""
    return Dataset(csv_path, transform=get_transform(split, input_size))


def get_dataloader(split, csv_path, input_size=512, batch_size=16, seed=42):
    """Return a Dataloader for the given split, built from csv_path with split-specific transform."""
    return Dataloader(
        split,
        dataset=Dataset(csv_path, transform=get_transform(split, input_size)),
        batch_size=batch_size,
        seed=seed,
    )


def get_samples(split, csv_path, input_size=512,
                indices=None, num_samples=None, shuffle=False, seed=42):
    """Return a batch of samples selected by indices or count as stacked tensors."""
    dataset = get_dataset(split, csv_path, input_size)
    if indices is not None:
        idx = indices
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
        if num_samples is not None:
            idx = idx[:num_samples]
    samples = [dataset[i] for i in idx]
    images, corners = zip(*samples)
    return torch.stack(list(images)), torch.stack(list(corners))
