"""Differentiable SSIM and combined MSE+SSIM losses for MRI super-resolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(
    window_size: int,
    sigma: float,
    channel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    window = g @ g.t()
    return window.expand(channel, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM (higher SSIM maps to lower loss)."""

    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        window = _gaussian_window(
            self.window_size, 1.5, channel, pred.device, pred.dtype
        )
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad, groups=channel)
        mu_t = F.conv2d(target, window, padding=pad, groups=channel)
        mu_p2, mu_t2, mu_pt = mu_p.pow(2), mu_t.pow(2), mu_p * mu_t

        sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu_p2
        sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channel) - mu_t2
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu_pt

        ssim_map = ((2 * mu_pt + self.c1) * (2 * sigma_pt + self.c2)) / (
            (mu_p2 + mu_t2 + self.c1) * (sigma_p2 + sigma_t2 + self.c2)
        )
        return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Weighted MSE + SSIM loss for perceptual super-resolution training."""

    def __init__(self, mse_weight: float = 1.0, ssim_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + self.ssim_weight * self.ssim(
            pred, target
        )
