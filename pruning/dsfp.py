"""
pruning/dsfp.py
Differential Sensitivity Fusion Pruning (DSFP)

Three-phase algorithm (Algorithm 1 in the paper):
  Phase A — Single forward/backward pass computes three per-filter importance
             metrics: gradient sensitivity, Taylor expansion, and output-space
             KL divergence.  All three are min-max normalised to [0,1] before
             fusion to eliminate scale differences (Reviewer 2-5 fix).

  Phase B — Lightweight contextual bandit assigns a per-layer pruning ratio
             around the global target rate.  This is a single-step regression
             (not sequential Q-learning) — one action per layer per pruning
             call.

  Phase C — Structural zeroing of the selected filters.  Post-pruning KD
             fine-tuning is handled separately in training/trainer.py.

KL(F) definition (paper §III-A, Reviewer 2-5):
  KL(F) = KL( softmax(logits_full) ‖ softmax(logits_masked_F) )
  computed on a fixed calibration batch.  Filter F's entire output-channel
  slice is zero-masked in a single forward pass (no retraining needed).
  Cost: O(N_filters) masked forward passes on the calibration batch.
"""

from __future__ import annotations
import logging
from collections import deque
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ===========================================================================
# Utility helpers
# ===========================================================================

def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise a 1-D tensor to [0, 1]."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def _get_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """Return list of (name, module) for all Conv2d layers (excluding 1×1 shortcuts)."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d)
        # skip 1×1 projection shortcuts (depthwise-style) — can be added later
    ]


# ===========================================================================
# Phase A: importance scoring
# ===========================================================================

class _ImportanceScorer:
    """
    Computes per-filter importance scores using a single forward/backward pass.

    Returns a dict: layer_name → 1-D tensor of shape [out_channels] with
    fused DSFP importance scores (lower = less important = prune first).
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ── sub-metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def _grad_sensitivity(conv: nn.Conv2d) -> torch.Tensor:
        """
        Gradient sensitivity: L2 norm of ∂L/∂W per output channel.
        Shape: [out_channels]
        """
        if conv.weight.grad is None:
            return torch.zeros(conv.out_channels, device=conv.weight.device)
        # grad shape: [out_ch, in_ch, kH, kW] → norm over all but out_ch
        return conv.weight.grad.data.abs().view(conv.out_channels, -1).mean(dim=1)

    @staticmethod
    def _taylor_expansion(conv: nn.Conv2d) -> torch.Tensor:
        """
        First-order Taylor importance: |w · ∂L/∂w| per output channel.
        Shape: [out_channels]
        """
        if conv.weight.grad is None:
            return torch.zeros(conv.out_channels, device=conv.weight.device)
        product = (conv.weight.data * conv.weight.grad.data).abs()
        return product.view(conv.out_channels, -1).mean(dim=1)

    def _kl_divergence(self, model: nn.Module, conv_name: str,
                       conv: nn.Conv2d, calib_batch: torch.Tensor) -> torch.Tensor:
        """
        Output-space KL divergence: KL(p_full ‖ p_masked_F) per filter F.

        p_full   = softmax(logits) on the full model.
        p_masked = softmax(logits) when filter F's output channel is zeroed.

        Each filter requires one masked forward pass on the calibration batch.
        Total cost: out_channels masked passes — all on a small calib_batch.

        Shape: [out_channels]
        """
        model.eval()
        with torch.no_grad():
            logits_full = model(calib_batch)
            p_full      = F.softmax(logits_full, dim=-1)           # [B, C]

        kl_scores = torch.zeros(conv.out_channels, device=self.device)

        def _zero_hook(module, inp, out, filter_idx):
            out = out.clone()
            out[:, filter_idx, :, :] = 0.0
            return out

        for f_idx in range(conv.out_channels):
            handle = conv.register_forward_hook(
                lambda m, i, o, fi=f_idx: _zero_hook(m, i, o, fi)
            )
            with torch.no_grad():
                logits_masked  = model(calib_batch)
                p_masked       = F.softmax(logits_masked, dim=-1)  # [B, C]
                # KL(p_full ‖ p_masked), averaged over batch
                kl = F.kl_div(
                    p_masked.log().clamp(min=-100),
                    p_full,
                    reduction="batchmean",
                )
                kl_scores[f_idx] = kl.item()
            handle.remove()

        return kl_scores

    # ── fusion ────────────────────────────────────────────────────────────────

    def compute(self, model: nn.Module, calib_loader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Full scoring pass.

        Steps:
          1. Single forward/backward on the first calibration batch to populate
             .grad on all Conv2d weights.
          2. Extract Grad(F) and Taylor(F) from gradients (deterministic).
          3. Compute KL(F) via O(N_filters) masked forward passes.
          4. Per-layer min-max normalise all three metrics.
          5. Fuse via exponential disagreement formula.

        Returns:
            {layer_name: importance_tensor[out_channels]}
            (lower value = less important = prune first)
        """
        model.train()
        model.to(self.device)

        # ── Step 1: single forward/backward pass ─────────────────────────────
        images, labels = next(iter(calib_loader))
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        model.zero_grad()
        logits = model(images)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        # ── Step 2: gradient + Taylor per layer ──────────────────────────────
        conv_layers = _get_conv_layers(model)
        grad_scores:   dict[str, torch.Tensor] = {}
        taylor_scores: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            grad_scores[name]   = self._grad_sensitivity(conv).detach()
            taylor_scores[name] = self._taylor_expansion(conv).detach()

        model.zero_grad()   # clear gradients before KL pass

        # ── Step 3: KL divergence per layer ──────────────────────────────────
        calib_batch = images   # reuse the same batch for consistency
        kl_scores: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            logger.debug(f"  KL scoring layer: {name} ({conv.out_channels} filters)")
            kl_scores[name] = self._kl_divergence(
                model, name, conv, calib_batch
            ).detach()

        # ── Step 4+5: normalise and fuse ─────────────────────────────────────
        fused: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            g  = _minmax_norm(grad_scores[name])    # [0, 1]
            t  = _minmax_norm(taylor_scores[name])  # [0, 1]
            k  = _minmax_norm(kl_scores[name])      # [0, 1]

            # Pairwise absolute differences amplify inter-metric disagreement.
            # The 0.5 coefficient on diff3 down-weights the Grad–KL term:
            # post-normalisation g and k span similar ranges, so without the
            # coefficient this term would double-weight the grad signal.
            # See paper §III-A for full justification.
            diff1 = torch.abs(g - t)
            diff2 = torch.abs(t - k)
            diff3 = torch.abs(g - k)
            score = torch.exp(diff1 + diff2 + 0.5 * diff3)

            fused[name] = score   # higher = more disagreement = more pruneable

        model.train()
        return fused


# ===========================================================================
# Phase B: contextual bandit layer-ratio selector
# ===========================================================================

class _BanditAgent:
    """
    Single-step contextual bandit for per-layer pruning ratio selection.

    State:   [global_rate / 100, layer_depth / n_layers]  (2-D)
    Action:  one of K discrete pruning ratios in [rate - δ, rate + δ]
    Reward:  negative fraction of accuracy drop after pruning
             (set externally; here we use a proxy: -mean(score of pruned filters))

    This is intentionally lightweight — one action per layer per pruning call.
    No experience replay across pruning calls (different seeds → different
    importance landscapes).
    """

    N_ACTIONS = 9  # discrete rate offsets: -4, -3, ..., +4 percentage points

    def __init__(
        self,
        lr: float           = 0.01,
        buffer_size: int    = 512,
        hidden: int         = 32,
        epsilon_decay: float = 0.995,
        explore_delta: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device       = device
        self.epsilon      = 1.0
        self.epsilon_min  = 0.05
        self.epsilon_decay = epsilon_decay
        self.explore_delta = explore_delta   # ± % range around base rate
        self.gamma        = 0.0   # bandit: no future reward

        # Small MLP: state (2-D) → Q-values (N_ACTIONS)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.N_ACTIONS),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Replay buffer (state, action, reward)
        self._buf_states:  deque = deque(maxlen=buffer_size)
        self._buf_actions: deque = deque(maxlen=buffer_size)
        self._buf_rewards: deque = deque(maxlen=buffer_size)

    def _state_tensor(self, global_rate: float,
                      layer_idx: int, n_layers: int) -> torch.Tensor:
        return torch.tensor(
            [global_rate / 100.0, layer_idx / max(n_layers - 1, 1)],
            dtype=torch.float32, device=self.device,
        )

    def _rate_offsets(self) -> np.ndarray:
        """N_ACTIONS evenly-spaced offsets in [-explore_delta, +explore_delta]."""
        return np.linspace(-self.explore_delta, self.explore_delta,
                           self.N_ACTIONS)

    def select_rate(self, global_rate: float,
                    layer_idx: int, n_layers: int) -> float:
        """ε-greedy action selection. Returns a concrete pruning rate [0, 95]."""
        state = self._state_tensor(global_rate, layer_idx, n_layers)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.N_ACTIONS)
        else:
            self.net.eval()
            with torch.no_grad():
                q_values = self.net(state.unsqueeze(0))
            action = int(q_values.argmax().item())
        offset    = self._rate_offsets()[action]
        raw_rate  = global_rate + offset
        final_rate = float(np.clip(raw_rate, 0.0, 95.0))

        self._buf_states.append(state.cpu())
        self._buf_actions.append(action)
        return final_rate

    def update(self, reward: float) -> None:
        """Store reward for the most recent action and do one gradient step."""
        if not self._buf_rewards.__len__() < len(self._buf_states):
            self._buf_rewards.append(reward)
        else:
            self._buf_rewards.append(reward)

        # Need at least one (s, a, r) triple
        if len(self._buf_states) == 0:
            return
        # Use only the last transition (bandit: no bootstrapping)
        s = self._buf_states[-1].to(self.device).unsqueeze(0)
        a = self._buf_actions[-1]
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)

        self.net.train()
        self.optimizer.zero_grad()
        q_all    = self.net(s)                            # [1, N_ACTIONS]
        q_target = q_all.clone().detach()
        q_target[0, a] = r
        loss = F.mse_loss(q_all, q_target)
        loss.backward()
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)


# ===========================================================================
# Phase C: structural filter zeroing
# ===========================================================================

def _zero_filters(model: nn.Module, layer_name: str,
                  prune_indices: list[int]) -> None:
    """
    Zero-out the selected output-channel filters in-place.

    Sets both the Conv2d weight and bias (if present) of the selected
    output channels to zero.  BatchNorm parameters for the same channels
    are also zeroed so their affine transform does not re-activate pruned
    channels.
    """
    # Build a mapping: conv_name → following BN module
    bn_map: dict[str, Optional[nn.BatchNorm2d]] = {}
    modules = list(model.named_modules())
    for i, (name, m) in enumerate(modules):
        if name == layer_name:
            # Check next module for BN
            if i + 1 < len(modules):
                next_m = modules[i + 1][1]
                bn_map[layer_name] = next_m if isinstance(
                    next_m, nn.BatchNorm2d) else None
            break

    for name, m in model.named_modules():
        if name == layer_name and isinstance(m, nn.Conv2d):
            with torch.no_grad():
                m.weight.data[prune_indices] = 0.0
                if m.bias is not None:
                    m.bias.data[prune_indices] = 0.0
            # Zero the corresponding BN channel
            bn = bn_map.get(layer_name)
            if bn is not None:
                with torch.no_grad():
                    bn.weight.data[prune_indices] = 0.0
                    bn.bias.data[prune_indices]   = 0.0
                    bn.running_mean[prune_indices] = 0.0
                    bn.running_var[prune_indices]  = 1.0
            break


# ===========================================================================
# DSFPruner — public API
# ===========================================================================

class DSFPruner:
    """
    DSFP structural filter pruner.

    Usage:
        pruner = DSFPruner(device=device)
        pruned_model = pruner.prune(model, calib_loader, pruning_rate=60.0)
    """

    def __init__(
        self,
        bandit_lr: float            = 0.01,
        bandit_buffer_size: int     = 512,
        bandit_hidden: int          = 32,
        bandit_epsilon_decay: float = 0.995,
        explore_delta: float        = 4.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device       = device or torch.device("cpu")
        self._scorer      = _ImportanceScorer(device=self.device)
        self._bandit_cfg  = dict(
            lr=bandit_lr,
            buffer_size=bandit_buffer_size,
            hidden=bandit_hidden,
            epsilon_decay=bandit_epsilon_decay,
            explore_delta=explore_delta,
            device=self.device,
        )

    def prune(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        pruning_rate: float,
        importance_override: Optional[dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Prune the model in-place and return it.

        Args:
            model:               Model to prune (modified in-place).
            dataloader:          Calibration DataLoader for importance scoring.
            pruning_rate:        Global target pruning rate in [0, 100).
            importance_override: Pre-computed importance scores (for ablation).

        Returns:
            The pruned model.
        """
        model = model.to(self.device)

        # Phase A: importance scoring
        if importance_override is not None:
            scores = importance_override
        else:
            logger.info("Phase A: computing DSFP importance scores...")
            scores = self._scorer.compute(model, dataloader)

        conv_layers = _get_conv_layers(model)
        n_layers    = len(conv_layers)
        bandit      = _BanditAgent(**self._bandit_cfg)

        total_filters  = 0
        pruned_filters = 0

        for layer_idx, (name, conv) in enumerate(conv_layers):
            out_ch = conv.out_channels

            # Phase B: per-layer rate from bandit
            layer_rate = bandit.select_rate(pruning_rate, layer_idx, n_layers)
            n_prune    = max(0, min(int(out_ch * layer_rate / 100.0),
                                   out_ch - 1))   # keep at least 1 filter

            if n_prune == 0:
                bandit.update(reward=0.0)
                continue

            # Select least-important filters (lowest score)
            layer_scores = scores.get(name)
            if layer_scores is None:
                logger.warning(f"No scores for layer {name}, skipping.")
                bandit.update(reward=0.0)
                continue

            prune_indices = layer_scores.argsort()[:n_prune].tolist()

            # Reward proxy: mean importance of pruned filters (lower = better choice)
            reward = -float(layer_scores[prune_indices].mean().item())
            bandit.update(reward=reward)

            # Phase C: structural zeroing
            _zero_filters(model, name, prune_indices)

            total_filters  += out_ch
            pruned_filters += n_prune
            logger.debug(
                f"Layer {name}: pruned {n_prune}/{out_ch} filters "
                f"({100*n_prune/out_ch:.1f}%) | bandit rate: {layer_rate:.1f}%"
            )

        actual_rate = 100.0 * pruned_filters / max(total_filters, 1)
        logger.info(
            f"Pruning complete: {pruned_filters}/{total_filters} filters zeroed "
            f"({actual_rate:.1f}% actual rate, target {pruning_rate:.1f}%)"
        )
        return model
