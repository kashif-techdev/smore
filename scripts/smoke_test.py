#!/usr/bin/env python3
"""Fast CI smoke tests — no dataset download or GPU training required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smore.losses import CombinedLoss, SSIMLoss  # noqa: E402
from smore.splits import subject_level_split  # noqa: E402


def test_subject_split_no_overlap() -> None:
    slices_by_subject = [[np.zeros((4, 4)) + i] for i in range(10)]
    split = subject_level_split(slices_by_subject, seed=0)
    assert len(split["train_slices"]) + len(split["val_slices"]) + len(split["test_slices"]) == 10
    assert split["train_subjects"] + split["val_subjects"] + split["test_subjects"] == 10
    ids = (
        set(split["train_subject_ids"])
        | set(split["val_subject_ids"])
        | set(split["test_subject_ids"])
    )
    assert ids == set(range(10))


def test_ssim_loss_decreases_when_matching() -> None:
    loss_fn = SSIMLoss()
    x = torch.rand(2, 1, 32, 32)
    same = SSIMLoss()(x, x).item()
    diff = loss_fn(x, torch.rand_like(x)).item()
    assert same < diff


def test_combined_loss_backward() -> None:
    model = torch.nn.Conv2d(1, 1, 3, padding=1)
    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.2)
    x = torch.rand(2, 1, 32, 32, requires_grad=False)
    y = torch.rand(2, 1, 32, 32)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    assert model.weight.grad is not None


def test_unet_forward_minimal() -> None:
    """Import U-Net definition from notebook is heavy; use a tiny surrogate."""
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
        torch.nn.Sigmoid(),
    )
    inp = torch.randn(1, 1, 128, 128)
    out = net(inp)
    assert out.shape == inp.shape


def main() -> None:
    test_subject_split_no_overlap()
    test_ssim_loss_decreases_when_matching()
    test_combined_loss_backward()
    test_unet_forward_minimal()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
