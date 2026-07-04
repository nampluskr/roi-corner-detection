# src/data/dataloader.py: batching iterator over a Dataset

import torch
from torch.utils.data import DataLoader as TorchDataLoader


class Dataloader(TorchDataLoader):
    """Batches (image, corners) samples from a Dataset."""

    def __init__(self, split, dataset, batch_size=16, seed=42):
        num_workers = 4 if split == "train" else 0
        generator = torch.Generator()
        generator.manual_seed(seed)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            generator=generator,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
