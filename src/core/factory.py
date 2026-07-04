# src/core/factory.py: factory functions for creating data pipeline objects

import torch
from src.data.dataset import CornerDataset, ImageDataset
from src.data.dataloader import Dataloader
from src.data.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    ColorJitter, GaussianBlur,
)


def get_transform(split, input_size=224):
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
        Normalize(),
    ]
    return Compose(transforms)


def get_dataset(split, csv_path, input_size=224, has_corners=True):
    """Return a CornerDataset or ImageDataset for the given split, built from csv_path."""
    dataset_cls = CornerDataset if has_corners else ImageDataset
    return dataset_cls(csv_path, transform=get_transform(split, input_size))


def get_dataloader(split, csv_path, input_size=224, batch_size=16, seed=42, has_corners=True):
    """Return a Dataloader for the given split, built from csv_path with split-specific transform."""
    return Dataloader(
        split,
        dataset=get_dataset(split, csv_path, input_size, has_corners=has_corners),
        batch_size=batch_size,
        seed=seed,
    )


def get_samples(split, csv_path, input_size=224, indices=None, num_samples=None,
                shuffle=False, seed=42, has_corners=True):
    """Return a batch of samples selected by indices or count as stacked tensors."""
    dataset = get_dataset(split, csv_path, input_size, has_corners=has_corners)

    if indices is not None:
        samples = [dataset[i] for i in indices]
    elif shuffle:
        sample_dataset = dataset.subset(num_samples if num_samples is not None else len(dataset), seed=seed)
        samples = [sample_dataset[i] for i in range(len(sample_dataset))]
    else:
        idx = list(range(len(dataset)))
        if num_samples is not None:
            idx = idx[:num_samples]
        samples = [dataset[i] for i in idx]

    if not has_corners:
        return torch.stack(samples)
    images, corners = zip(*samples)
    return torch.stack(list(images)), torch.stack(list(corners))
