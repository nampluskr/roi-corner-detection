# src/data/dataset.py: dataset that loads corner coordinates from a CSV file

import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, random_split

from src.data.transforms import ToTensor


class Dataset(TorchDataset):
    """Loads (image, corners) pairs from a gt_corners.csv-format file."""

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
                for row in csv.DictReader(f):
                    image_path = os.path.join(row["image_dir"], row["image_name"])
                    corners = np.array([
                        row["x1"], row["y1"], row["x2"], row["y2"],
                        row["x3"], row["y3"], row["x4"], row["y4"],
                    ], dtype=np.float32).reshape(4, 2)
                    samples.append((image_path, corners))
        return samples

    def split(self, split_ratio=0.8, seed=42):
        n_train = int(len(self) * split_ratio)
        n_valid = len(self) - n_train
        generator = torch.Generator().manual_seed(seed)
        return random_split(self, [n_train, n_valid], generator=generator)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, corners = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image, corners = self.transform(image, corners)
        return image, corners
