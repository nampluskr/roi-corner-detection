# src/data/dataloader.py: batching iterator over a dataset

import torch


class Dataloader(torch.utils.data.DataLoader):
    """Batches samples from a dataset; batch shape follows the dataset's __getitem__."""

    def __init__(self, split, dataset, batch_size=16, seed=42, num_workers=None):
        if num_workers is None:
            num_workers = 4 if split == "train" else 0
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            generator=generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs,
        )
