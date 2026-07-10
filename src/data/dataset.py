# src/data/dataset.py: datasets that load image paths, with or without corner coordinates, from CSV files

import os
import csv
import copy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split, Dataset as TorchDataset, Subset as TorchSubset

from src.data.transforms import ToTensor


class BaseDataset(TorchDataset):
    """Shared splitting and subsetting logic for CSV-backed datasets and their subsets."""

    def subset(self, num_samples, seed=42):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self), generator=generator)[:num_samples].tolist()
        return Subset(self, indices)

    def split(self, split_ratio=0.8, seed=42):
        n_first = int(len(self) * split_ratio)
        n_second = len(self) - n_first
        generator = torch.Generator().manual_seed(seed)
        first_indices, second_indices = random_split(
            range(len(self)), [n_first, n_second], generator=generator)
        return Subset(self, list(first_indices)), Subset(self, list(second_indices))


class Dataset(BaseDataset):
    """Loads CSV-backed samples and rebuilds itself with a new transform."""

    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform or ToTensor()
        self.samples = self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        if isinstance(csv_path, str):
            csv_path = [csv_path]
        samples = []
        for p in csv_path:
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(self._parse_row(row))
        return samples

    def _parse_row(self, row):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError

    def set_transform(self, transform):
        new_dataset = copy.copy(self)
        new_dataset.transform = transform
        return new_dataset


class Subset(BaseDataset):
    """Wraps a torch Subset so set_transform can rebuild it with a new transform."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self._subset = TorchSubset(dataset, indices)

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]

    def set_transform(self, transform):
        new_dataset = self.dataset.set_transform(transform)
        return Subset(new_dataset, self.indices)


class CornerDataset(Dataset):
    """Loads (image, corners) pairs from a CSV with x1..y4 corner columns."""

    def _parse_row(self, row):
        image_path = os.path.join(row["image_dir"], row["image_name"])
        corners = np.array([
            row["x1"], row["y1"], row["x2"], row["y2"],
            row["x3"], row["y3"], row["x4"], row["y4"],
        ], dtype=np.float32).reshape(4, 2)
        return image_path, corners

    def __getitem__(self, idx):
        image_path, corners = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image, corners = self.transform(image, corners)
        return image, corners


class ImageDataset(Dataset):
    """Loads bare images from a CSV with no corner columns."""

    def _parse_row(self, row):
        return os.path.join(row["image_dir"], row["image_name"])

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
