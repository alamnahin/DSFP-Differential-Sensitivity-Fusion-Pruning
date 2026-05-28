"""
utils/data.py
Unified data-loading for CIFAR-10, CIFAR-100, and Tiny-ImageNet.

Returns:
    train_loader  – augmented training set
    test_loader   – clean test / validation set
    calib_loader  – fixed calibration subset (for DSFP importance scoring)

The calibration loader draws `calibration_size` samples from the *training*
set (without augmentation) so importance scores are computed on clean,
in-distribution data. The same samples are used across all pruning rates
within one seed for fair comparison.
"""

from __future__ import annotations
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-dataset normalization statistics
# ---------------------------------------------------------------------------

_STATS = {
    "cifar10":       {"mean": (0.4914, 0.4822, 0.4465),
                      "std":  (0.2023, 0.1994, 0.2010)},
    "cifar100":      {"mean": (0.5071, 0.4867, 0.4408),
                      "std":  (0.2675, 0.2565, 0.2761)},
    "tiny-imagenet": {"mean": (0.4802, 0.4481, 0.3975),
                      "std":  (0.2770, 0.2691, 0.2821)},
}


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _train_transform(dataset: str) -> transforms.Compose:
    stats  = _STATS[dataset]
    norm   = transforms.Normalize(stats["mean"], stats["std"])
    size   = 64 if dataset == "tiny-imagenet" else 32
    crop   = 64 if dataset == "tiny-imagenet" else 32
    pad    = 8  if dataset == "tiny-imagenet" else 4
    return transforms.Compose([
        transforms.RandomCrop(crop, padding=pad),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        norm,
    ])


def _test_transform(dataset: str) -> transforms.Compose:
    stats = _STATS[dataset]
    norm  = transforms.Normalize(stats["mean"], stats["std"])
    return transforms.Compose([transforms.ToTensor(), norm])


def _calib_transform(dataset: str) -> transforms.Compose:
    """No augmentation for calibration — deterministic importance scores."""
    return _test_transform(dataset)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_dataset(dataset: str, data_dir: str,
                   train: bool, transform: transforms.Compose):
    os.makedirs(data_dir, exist_ok=True)
    name = dataset.lower()
    if name == "cifar10":
        return datasets.CIFAR10(data_dir, train=train,
                                transform=transform, download=True)
    if name == "cifar100":
        return datasets.CIFAR100(data_dir, train=train,
                                 transform=transform, download=True)
    if name == "tiny-imagenet":
        split = "train" if train else "val"
        root  = os.path.join(data_dir, "tiny-imagenet-200", split)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {root}. "
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to data_dir."
            )
        return datasets.ImageFolder(root, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    calibration_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / test / calibration DataLoaders.

    Args:
        dataset:          One of cifar10 | cifar100 | tiny-imagenet.
        data_dir:         Root directory for dataset storage.
        batch_size:       Mini-batch size for train and test.
        num_workers:      DataLoader worker threads.
        calibration_size: Number of training samples for calibration subset.
        seed:             Random seed for reproducible calibration split.

    Returns:
        (train_loader, test_loader, calib_loader)
    """
    name = dataset.lower()
    if name not in _STATS:
        raise ValueError(f"Dataset '{dataset}' not supported. "
                         f"Choose from: {list(_STATS.keys())}")

    # ── Full training set (augmented) ─────────────────────────────────────────
    train_ds = _build_dataset(name, data_dir, train=True,
                              transform=_train_transform(name))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    # ── Test set ──────────────────────────────────────────────────────────────
    test_ds = _build_dataset(name, data_dir, train=False,
                             transform=_test_transform(name))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ── Calibration subset (no augmentation, seeded) ─────────────────────────
    calib_ds = _build_dataset(name, data_dir, train=True,
                              transform=_calib_transform(name))
    rng      = np.random.default_rng(seed)
    n_train  = len(calib_ds)
    indices  = rng.choice(n_train, size=min(calibration_size, n_train),
                          replace=False).tolist()
    calib_subset  = Subset(calib_ds, indices)
    calib_loader  = DataLoader(
        calib_subset,
        batch_size=min(calibration_size, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Dataset: {dataset} | "
        f"Train: {len(train_ds)} | Test: {len(test_ds)} | "
        f"Calib: {len(calib_subset)} (seed={seed})"
    )
    return train_loader, test_loader, calib_loader
