# src/core/factory.py: factory functions for creating shared modules and objects

import os
import logging
import torch
from src.data.dataset import CornerDataset, ImageDataset
from src.data.dataloader import Dataloader
from src.data.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    ColorJitter, GaussianBlur,
)


def get_transform(split, image_size=224):
    """Return a Compose of (image, corners) transforms for the given split."""
    transforms = [Resize((image_size, image_size))]
    if split == "train":
        transforms += [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=5.0),
            ColorJitter(brightness=0.2, contrast=0.2),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]
    transforms += [ToTensor(), Normalize()]
    return Compose(transforms)


def get_dataset(split, csv_path, image_size=224, has_corners=True, split_ratio=0.6, seed=42):
    """Return the train, valid, or test split of a CornerDataset/ImageDataset with split-specific transform."""
    dataset_cls = CornerDataset if has_corners else ImageDataset
    dataset = dataset_cls(csv_path)
    train_dataset, temp_dataset = dataset.split(split_ratio, seed)
    valid_dataset, test_dataset = temp_dataset.split(0.5, seed)
    subsets = {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
    subset = subsets[split]
    return subset.set_transform(get_transform(split, image_size))


def get_dataloader(split, csv_path, image_size=224, has_corners=True, split_ratio=0.6, seed=42,
                    batch_size=32, num_workers=None, num_samples=None):
    """Return a Dataloader for the given split, built from csv_path with split-specific transform."""
    dataset = get_dataset(split, csv_path, image_size=image_size,
                          has_corners=has_corners, split_ratio=split_ratio, seed=seed)
    if num_samples is not None:
        dataset = dataset.subset(num_samples, seed=seed)
    return Dataloader(split, dataset, batch_size=batch_size, seed=seed, num_workers=num_workers)


def get_samples(split, csv_path, image_size=224, has_corners=True, split_ratio=0.6, seed=42,
                indices=None, num_samples=None, shuffle=False):
    """Return a batch of samples selected by indices or count as stacked tensors."""
    dataset = get_dataset(split, csv_path, image_size=image_size,
                          has_corners=has_corners, split_ratio=split_ratio, seed=seed)

    if indices is not None:
        samples = [dataset[i] for i in indices]
    elif shuffle:
        subset = dataset.subset(num_samples if num_samples is not None else len(dataset), seed=seed)
        samples = [subset[i] for i in range(len(subset))]
    else:
        idx = list(range(len(dataset)))
        if num_samples is not None:
            idx = idx[:num_samples]
        samples = [dataset[i] for i in idx]

    if not has_corners:
        return torch.stack(samples)
    images, corners = zip(*samples)
    return torch.stack(list(images)), torch.stack(list(corners))


def get_wrapper(method, device=None, **kwargs):
    """Return a method-specific wrapper built with the given device and kwargs."""
    if method == "direct":
        from src.models.direct.wrapper import DirectWrapper
        return DirectWrapper(device=device, **kwargs)
    elif method == "homography":
        from src.models.homography.wrapper import HomographyWrapper
        return HomographyWrapper(device=device, **kwargs)
    elif method == "heatmap":
        from src.models.heatmap.wrapper import HeatmapWrapper
        return HeatmapWrapper(device=device, **kwargs)
    raise NotImplementedError("method not yet implemented: %s" % method)


def get_logger(name, output_dir=None):
    """Return a logger with plain terminal output and, if output_dir is set, a timestamped output_dir/run.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    stream_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
