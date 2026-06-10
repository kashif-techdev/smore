"""SSIM + MSE combined loss for MRI super-resolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_metric


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """Weighted loss: alpha * (1 - SSIM) + (1 - alpha) * MSE."""
    mse_val = F.mse_loss(pred, target)
    ssim_val = 1.0 - ssim_metric(pred, target, data_range=1.0, size_average=True)
    return alpha * ssim_val + (1.0 - alpha) * mse_val


class CombinedLoss(nn.Module):
    """Module wrapper around combined_loss for training loops."""

    def __init__(self, ssim_alpha: float = 0.8):
        super().__init__()
        self.ssim_alpha = ssim_alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return combined_loss(pred, target, alpha=self.ssim_alpha)
