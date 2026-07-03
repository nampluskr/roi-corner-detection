# src/data/dataloader.py: batching iterator over a CornerDataset
import torch
from torch.utils.data import DataLoader as TorchDataloader


class Dataloader:
    """Batches (image, corners) samples from a CornerDataset."""

    def __init__(self, split, dataset, batch_size=16, seed=42):
        self.split = split
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        generator = torch.Generator().manual_seed(seed)
        self.loader = TorchDataloader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            generator=generator,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)
