"""Train/validation splits and target statistics."""

from __future__ import annotations

import torch
from torch.utils.data import Subset, random_split


def random_train_val_split(dataset, val_fraction: float = 0.2, seed: int = 42):
    """Random train/val split. Returns (train_subset, val_subset)."""
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def compute_target_stats(dataset, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of log1p(target) over dataset indices.

    Returns (y_mean, y_std) as scalar tensors.
    """
    ys = []
    idxs = indices if indices is not None else range(len(dataset))
    for i in idxs:
        data = dataset[i]
        if hasattr(data, "y") and data.y is not None:
            ys.append(data.y.reshape(-1))
    log_ys = torch.log1p(torch.cat(ys))
    return log_ys.mean(), log_ys.std()
