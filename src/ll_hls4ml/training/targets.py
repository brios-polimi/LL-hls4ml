"""Target normalization for log-transformed LUT (or other) labels."""

import torch


def normalize_target(y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return (torch.log1p(y) - y_mean) / y_std


def to_luts(pred: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    """Denormalize model output back to original LUT scale."""
    return torch.expm1(pred * y_std + y_mean)
