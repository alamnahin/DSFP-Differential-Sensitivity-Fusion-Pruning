"""
utils/metrics.py
Evaluation utilities for accuracy, FLOPs, parameter counts, and filter counts.

FLOPs are computed with the `thop` library (pip install thop), which gives
per-operator counts consistent with the paper's reported MFLOPs.  Results are
always returned in **MFLOPs** (10^6 FLOPs) with explicit units logged.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy(model: nn.Module,
                     loader: DataLoader,
                     device: torch.device) -> float:
    """Top-1 accuracy (%) over the full loader."""
    model.eval()
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), \
                         labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# FLOPs via thop
# ---------------------------------------------------------------------------

def compute_flops(model: nn.Module,
                  input_size: tuple = (1, 3, 32, 32),
                  device: Optional[torch.device] = None) -> float:
    """
    Compute model FLOPs in MFLOPs using the `thop` library.

    Args:
        model:      PyTorch model (eval mode recommended).
        input_size: Input tensor shape (batch, C, H, W).
        device:     Device for the dummy input.

    Returns:
        FLOPs in MFLOPs (float).
    """
    try:
        from thop import profile, clever_format  # type: ignore
    except ImportError:
        logger.warning("thop not installed. Run `pip install thop`. "
                       "Returning 0 MFLOPs.")
        return 0.0

    dev   = device or torch.device("cpu")
    dummy = torch.zeros(*input_size).to(dev)
    model.eval()
    model.to(dev)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    # thop returns MACs; FLOPs ≈ 2 × MACs for conv/linear layers
    flops_m = 2 * macs / 1e6
    logger.debug(f"FLOPs: {flops_m:.2f} MFLOPs")
    return flops_m


def get_input_size_for_dataset(dataset: str) -> tuple:
    """Return (1, C, H, W) matching the dataset's spatial resolution."""
    if dataset in ("cifar10", "cifar100"):
        return (1, 3, 32, 32)
    if dataset == "tiny-imagenet":
        return (1, 3, 64, 64)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_parameters(model: nn.Module) -> int:
    """Number of non-zero trainable parameters (after pruning masks applied)."""
    return sum((p != 0).sum().item()
               for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Filter (channel) counts
# ---------------------------------------------------------------------------

def count_nonzero_filters(model: nn.Module) -> int:
    """
    Count filters (output channels) whose weight tensors are not all-zero.

    A filter is considered pruned (zero) when the entire output-channel slice
    of its Conv2d weight tensor is identically zero.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # shape: [out_channels, in_channels, kH, kW]
            norms = m.weight.data.abs().view(m.weight.size(0), -1).sum(dim=1)
            total += int((norms > 0).sum().item())
    return total


def count_total_filters(model: nn.Module) -> int:
    """Count total output channels across all Conv2d layers."""
    return sum(m.out_channels
               for m in model.modules() if isinstance(m, nn.Conv2d))


# ---------------------------------------------------------------------------
# Sparsity summary
# ---------------------------------------------------------------------------

def sparsity_summary(model: nn.Module) -> dict:
    """Return a dict with param/filter sparsity statistics."""
    total_params   = count_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    total_filters  = count_total_filters(model)
    nonzero_filt   = count_nonzero_filters(model)
    return {
        "total_params":          total_params,
        "nonzero_params":        nonzero_params,
        "param_sparsity_pct":    round(100.0 * (1 - nonzero_params / max(total_params, 1)), 2),
        "total_filters":         total_filters,
        "nonzero_filters":       nonzero_filt,
        "filter_sparsity_pct":   round(100.0 * (1 - nonzero_filt / max(total_filters, 1)), 2),
    }
