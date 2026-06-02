"""Subject-level dataset splits for medical imaging."""

from __future__ import annotations

from typing import Any

import numpy as np


def subject_level_split(
    slices_by_subject: list[list[Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Split volumes by subject so no patient appears in more than one partition.

    Returns flattened slice lists plus subject counts per split.
    """
    n_subjects = len(slices_by_subject)
    if n_subjects < 3:
        raise ValueError("Need at least 3 subjects for train/val/test splits.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_subjects)

    n_train = max(1, int(train_ratio * n_subjects))
    n_val = max(1, int(val_ratio * n_subjects))
    if n_train + n_val >= n_subjects:
        n_val = max(1, n_subjects - n_train - 1)

    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val :]

    def gather(indices: np.ndarray) -> list[Any]:
        out: list[Any] = []
        for idx in indices:
            out.extend(slices_by_subject[int(idx)])
        return out

    return {
        "train_slices": gather(train_ids),
        "val_slices": gather(val_ids),
        "test_slices": gather(test_ids),
        "train_subjects": len(train_ids),
        "val_subjects": len(val_ids),
        "test_subjects": len(test_ids),
        "train_subject_ids": train_ids.tolist(),
        "val_subject_ids": val_ids.tolist(),
        "test_subject_ids": test_ids.tolist(),
    }
