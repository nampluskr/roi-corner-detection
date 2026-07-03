# src/data/dataset.py: dataset that loads corner coordinates from a CSV file
import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class CornerDataset(TorchDataset):
    """Loads (image, corners) pairs from a gt_corners.csv-format file."""

    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.df = pd.read_csv(csv_path)

    def split(self, split_ratio=0.8, seed=42):
        indices = list(range(len(self.df)))
        random.Random(seed).shuffle(indices)
        split_idx = int(len(indices) * split_ratio)
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

        train_dataset = CornerDataset.__new__(CornerDataset)
        train_dataset.csv_path = self.csv_path
        train_dataset.transform = self.transform
        train_dataset.df = self.df.iloc[train_indices].reset_index(drop=True)

        valid_dataset = CornerDataset.__new__(CornerDataset)
        valid_dataset.csv_path = self.csv_path
        valid_dataset.transform = self.transform
        valid_dataset.df = self.df.iloc[valid_indices].reset_index(drop=True)

        return train_dataset, valid_dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(row["image_dir"], row["image_name"])
        image = Image.open(image_path).convert("RGB")
        corners = row[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].to_numpy(dtype="float32").reshape(4, 2)

        if self.transform is not None:
            image, corners = self.transform(image, corners)

        return image, corners
