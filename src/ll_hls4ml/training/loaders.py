"""PyG DataLoader helpers."""

import os

from torch_geometric.loader import DataLoader as PyGDataLoader


def make_loader(ds, batch_size, shuffle=True):
    cpu_cores = os.cpu_count() or 2
    num_workers = max(2, min(4, cpu_cores))

    return PyGDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
