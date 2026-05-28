"""
pruning/baselines.py
Reproduced baseline pruning methods, each sharing the same
.prune(model, dataloader, pruning_rate) interface as DSFPruner.

Methods
-------
L1NormPruner  : Li et al. (2017) — prune filters with smallest L1-norm weight.
TaylorPruner  : Molchanov et al. (2019) — first-order Taylor expansion only
                (no KL, no fusion).  Uses same single-pass grad as DSFP Phase A.
SNIPPruner    : Lee et al. (2019) — single-shot gradient-based; scores
                connection sensitivity |g_i · w_i| on one calibration batch.
RandomPruner  : Randomly selects filters to prune (seeded via torch RNG).
                Used as the lower-bound sanity baseline.

All methods apply the same structural zeroing used by DSFP (_zero_filters)
so post-pruning KD fine-tuning is directly comparable.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pruning.dsfp import _get_conv_layers, _zero_filters

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _uniform_prune(model: nn.Module, scores: dict[str, torch.Tensor],
                   pruning_rate: float) -> nn.Module:
    """
    Apply uniform per-layer pruning at `pruning_rate`% using pre-computed
    per-filter importance scores (lower score = prune first).
    """
    for name, conv in _get_conv_layers(model):
        layer_scores = scores.get(name)
        if layer_scores is None:
            continue
        out_ch  = conv.out_channels
        n_prune = max(0, min(int(out_ch * pruning_rate / 100.0), out_ch - 1))
        if n_prune == 0:
            continue
        prune_indices = layer_scores.argsort()[:n_prune].tolist()
        _zero_filters(model, name, prune_indices)
        logger.debug(f"[baseline] Layer {name}: pruned {n_prune}/{out_ch} filters")
    return model


def _single_grad_pass(model: nn.Module, dataloader: DataLoader,
                      device: torch.device) -> None:
    """Run one forward/backward pass to populate .grad on Conv2d weights."""
    model.train()
    images, labels = next(iter(dataloader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    model.zero_grad()
    loss = F.cross_entropy(model(images), labels)
    loss.backward()


# ---------------------------------------------------------------------------
# L1-Norm Pruner
# ---------------------------------------------------------------------------

class L1NormPruner:
    """
    Prune filters with the smallest L1-norm of their weight tensor.
    Reference: Li et al., "Pruning Filters for Efficient ConvNets", ICLR 2017.
    No forward pass required — purely weight-magnitude based.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def prune(self, model: nn.Module, dataloader: DataLoader,
              pruning_rate: float) -> nn.Module:
        model = model.to(self.device)
        scores: dict[str, torch.Tensor] = {}
        for name, conv in _get_conv_layers(model):
            # L1-norm of each filter: sum |w| over [in_ch, kH, kW]
            norms = conv.weight.data.abs().view(conv.out_channels, -1).sum(dim=1)
            scores[name] = norms   # lower = prune first
        logger.info(f"L1NormPruner: pruning at {pruning_rate:.1f}%")
        return _uniform_prune(model, scores, pruning_rate)


# ---------------------------------------------------------------------------
# Taylor Pruner (single-metric)
# ---------------------------------------------------------------------------

class TaylorPruner:
    """
    Taylor-expansion-only pruning: |w · ∂L/∂w| per filter, averaged over
    [in_ch, kH, kW].  Single forward/backward pass on calibration batch.
    Reference: Molchanov et al., NeurIPS 2019.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def prune(self, model: nn.Module, dataloader: DataLoader,
              pruning_rate: float) -> nn.Module:
        model = model.to(self.device)
        _single_grad_pass(model, dataloader, self.device)

        scores: dict[str, torch.Tensor] = {}
        for name, conv in _get_conv_layers(model):
            if conv.weight.grad is None:
                scores[name] = torch.zeros(conv.out_channels,
                                           device=self.device)
                continue
            taylor = (conv.weight.data * conv.weight.grad.data).abs()
            scores[name] = taylor.view(conv.out_channels, -1).mean(dim=1)

        model.zero_grad()
        logger.info(f"TaylorPruner: pruning at {pruning_rate:.1f}%")
        return _uniform_prune(model, scores, pruning_rate)


# ---------------------------------------------------------------------------
# SNIP Pruner
# ---------------------------------------------------------------------------

class SNIPPruner:
    """
    Single-shot Network Pruning based on connection sensitivity.
    Score(w_i) = |g_i · w_i| / sum_j |g_j · w_j| (per layer, normalised).
    Reference: Lee et al., "SNIP: Single-shot Network Pruning", ICLR 2019.

    Here we apply SNIP at the filter (output-channel) level rather than
    individual connection level, consistent with our structural pruning setup.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def prune(self, model: nn.Module, dataloader: DataLoader,
              pruning_rate: float) -> nn.Module:
        model = model.to(self.device)
        _single_grad_pass(model, dataloader, self.device)

        scores: dict[str, torch.Tensor] = {}
        for name, conv in _get_conv_layers(model):
            if conv.weight.grad is None:
                scores[name] = torch.zeros(conv.out_channels,
                                           device=self.device)
                continue
            sensitivity = (conv.weight.data * conv.weight.grad.data).abs()
            per_filter  = sensitivity.view(conv.out_channels, -1).sum(dim=1)
            # Normalise within layer so scores are comparable across layers
            denom       = per_filter.sum() + 1e-8
            scores[name] = per_filter / denom   # lower = less sensitive = prune

        model.zero_grad()
        logger.info(f"SNIPPruner: pruning at {pruning_rate:.1f}%")
        return _uniform_prune(model, scores, pruning_rate)


# ---------------------------------------------------------------------------
# Random Pruner (sanity / lower bound baseline)
# ---------------------------------------------------------------------------

class RandomPruner:
    """
    Randomly select filters to prune (uniform, seeded via PyTorch RNG).
    Serves as the lower-bound sanity baseline to confirm that DSFP's accuracy
    recovery is not solely due to KD fine-tuning.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def prune(self, model: nn.Module, dataloader: DataLoader,
              pruning_rate: float) -> nn.Module:
        model = model.to(self.device)
        for name, conv in _get_conv_layers(model):
            out_ch  = conv.out_channels
            n_prune = max(0, min(int(out_ch * pruning_rate / 100.0),
                                 out_ch - 1))
            if n_prune == 0:
                continue
            perm          = torch.randperm(out_ch)
            prune_indices = perm[:n_prune].tolist()
            _zero_filters(model, name, prune_indices)
        logger.info(f"RandomPruner: pruned {pruning_rate:.1f}% filters (random)")
        return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PRUNER_MAP = {
    "l1norm": L1NormPruner,
    "taylor": TaylorPruner,
    "snip":   SNIPPruner,
    "random": RandomPruner,
}


def get_baseline_pruner(method: str,
                        device: Optional[torch.device] = None):
    """Return an instantiated baseline pruner by name."""
    method = method.lower()
    if method not in _PRUNER_MAP:
        raise ValueError(f"Unknown baseline method '{method}'. "
                         f"Choose from: {list(_PRUNER_MAP.keys())}")
    return _PRUNER_MAP[method](device=device)
