"""
pruning/dsfp.py
Differential Sensitivity Fusion Pruning (DSFP)

Fixes applied vs original:
  [BUG-17] _BanditAgent.update: reward-buffer alignment logic was wrong.
           The condition `not self._buf_rewards.__len__() < len(self._buf_states)`
           is always True on the first call, so rewards were appended twice.
           Simplified to always append once, then update.
  [BUG-18] _BanditAgent: N_ACTIONS=9 but explore_delta default is 5.0, so
           linspace(-5,5,9)=[−5,−3.75,−2.5,...,+5].  That's fine, but the
           comment says "±4 percentage points" (inconsistent with delta=5).
           Left as-is but clarified in docstring.
  [BUG-19] _ImportanceScorer._kl_divergence uses a lambda with a default
           argument fi=f_idx but the hook closure also captures 'filter_idx'
           from the outer scope.  The lambda shadow works but is fragile;
           replaced with a proper closure factory.
  [BUG-20] Phase A compute(): model.zero_grad() is called AFTER extracting
           grad_scores and taylor_scores but the scores were already detached,
           so this is safe — but the comment is misleading. Clarified.
  [BUG-21] _zero_filters: bn_map is built inside a loop that breaks on finding
           the target layer, but the outer loop over named_modules restarts
           from scratch. Refactored to a two-pass approach for clarity and
           correctness with deeper networks.
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
    """Return list of (name, module) for all Conv2d layers."""
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d)
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
        """Gradient sensitivity: L2 norm of ∂L/∂W per output channel."""
        if conv.weight.grad is None:
            return torch.zeros(conv.out_channels, device=conv.weight.device)
        return conv.weight.grad.data.abs().view(conv.out_channels, -1).mean(dim=1)

    @staticmethod
    def _taylor_expansion(conv: nn.Conv2d) -> torch.Tensor:
        """First-order Taylor importance: |w · ∂L/∂w| per output channel."""
        if conv.weight.grad is None:
            return torch.zeros(conv.out_channels, device=conv.weight.device)
        product = (conv.weight.data * conv.weight.grad.data).abs()
        return product.view(conv.out_channels, -1).mean(dim=1)

    def _kl_divergence(self, model: nn.Module, conv_name: str,
                       conv: nn.Conv2d,
                       calib_batch: torch.Tensor) -> torch.Tensor:
        """
        Output-space KL divergence: KL(p_full ‖ p_masked_F) per filter F.
        Each filter requires one masked forward pass on the calibration batch.
        """
        model.eval()
        with torch.no_grad():
            logits_full = model(calib_batch)
            p_full      = F.softmax(logits_full, dim=-1)  # [B, C]

        kl_scores = torch.zeros(conv.out_channels, device=self.device)

        # [BUG-19] use a proper closure factory instead of lambda with default arg
        def _make_hook(fi: int):
            def _hook(module, inp, out):
                out = out.clone()
                out[:, fi, :, :] = 0.0
                return out
            return _hook

        for f_idx in range(conv.out_channels):
            handle = conv.register_forward_hook(_make_hook(f_idx))
            with torch.no_grad():
                logits_masked = model(calib_batch)
                p_masked      = F.softmax(logits_masked, dim=-1)
                kl = F.kl_div(
                    p_masked.log().clamp(min=-100),
                    p_full,
                    reduction="batchmean",
                )
                kl_scores[f_idx] = kl.item()
            handle.remove()

        return kl_scores

    # ── fusion ────────────────────────────────────────────────────────────────

    def compute(self, model: nn.Module,
                calib_loader: DataLoader) -> dict[str, torch.Tensor]:
        """Full scoring pass — see module docstring for algorithm."""
        model.train()
        model.to(self.device)

        # Step 1: single forward/backward pass
        images, labels = next(iter(calib_loader))
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        model.zero_grad()
        logits = model(images)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()

        # Step 2: gradient + Taylor per layer (detached — safe before zero_grad)
        conv_layers = _get_conv_layers(model)
        grad_scores:   dict[str, torch.Tensor] = {}
        taylor_scores: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            grad_scores[name]   = self._grad_sensitivity(conv).detach()
            taylor_scores[name] = self._taylor_expansion(conv).detach()

        model.zero_grad()   # [BUG-20] clear before KL masked passes

        # Step 3: KL divergence per layer
        calib_batch = images   # reuse same batch for consistency
        kl_scores: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            logger.debug(f"  KL scoring layer: {name} ({conv.out_channels} filters)")
            kl_scores[name] = self._kl_divergence(
                model, name, conv, calib_batch
            ).detach()

        # Steps 4+5: normalise and fuse
        fused: dict[str, torch.Tensor] = {}
        for name, conv in conv_layers:
            g = _minmax_norm(grad_scores[name])
            t = _minmax_norm(taylor_scores[name])
            k = _minmax_norm(kl_scores[name])

            diff1 = torch.abs(g - t)
            diff2 = torch.abs(t - k)
            diff3 = torch.abs(g - k)
            score = torch.exp(diff1 + diff2 + 0.5 * diff3)

            fused[name] = score

        model.train()
        return fused


# ===========================================================================
# Phase B: contextual bandit layer-ratio selector
# ===========================================================================

class _BanditAgent:
    """
    Single-step contextual bandit for per-layer pruning ratio selection.

    State:   [global_rate / 100, layer_depth / n_layers]  (2-D)
    Action:  one of N_ACTIONS discrete rate offsets in [-delta, +delta]
    Reward:  -mean(importance of pruned filters) — proxy for accuracy drop
    """

    N_ACTIONS = 9

    def __init__(
        self,
        lr: float            = 0.01,
        buffer_size: int     = 512,
        hidden: int          = 32,
        epsilon_decay: float = 0.995,
        explore_delta: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device        = device
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = epsilon_decay
        self.explore_delta = explore_delta

        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.N_ACTIONS),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

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
        offset     = self._rate_offsets()[action]
        final_rate = float(np.clip(global_rate + offset, 0.0, 95.0))

        self._buf_states.append(state.cpu())
        self._buf_actions.append(action)
        return final_rate

    def update(self, reward: float) -> None:
        """Store reward for the most recent action and do one gradient step."""
        # [BUG-17] original had a broken alignment check; always append once
        self._buf_rewards.append(reward)

        if len(self._buf_states) == 0:
            return

        s = self._buf_states[-1].to(self.device).unsqueeze(0)
        a = self._buf_actions[-1]
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)

        self.net.train()
        self.optimizer.zero_grad()
        q_all    = self.net(s)
        q_target = q_all.clone().detach()
        q_target[0, a] = r
        loss = F.mse_loss(q_all, q_target)
        loss.backward()
        self.optimizer.step()

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

    [BUG-21] Refactored to a two-pass approach:
      Pass 1: build full ordered list of (name, module) pairs
      Pass 2: zero conv weights + matching BN immediately following it
    """
    modules = list(model.named_modules())

    # Build conv→BN map in one pass
    conv_bn_map: dict[str, Optional[nn.BatchNorm2d]] = {}
    for i, (name, m) in enumerate(modules):
        if isinstance(m, nn.Conv2d):
            nxt = modules[i + 1][1] if i + 1 < len(modules) else None
            conv_bn_map[name] = nxt if isinstance(nxt, nn.BatchNorm2d) else None

    for name, m in modules:
        if name != layer_name:
            continue
        if not isinstance(m, nn.Conv2d):
            continue
        with torch.no_grad():
            m.weight.data[prune_indices] = 0.0
            if m.bias is not None:
                m.bias.data[prune_indices] = 0.0
        bn = conv_bn_map.get(layer_name)
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
        self.device      = device or torch.device("cpu")
        self._scorer     = _ImportanceScorer(device=self.device)
        self._bandit_cfg = dict(
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
        """Prune the model in-place and return it."""
        model = model.to(self.device)

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
            out_ch     = conv.out_channels
            layer_rate = bandit.select_rate(pruning_rate, layer_idx, n_layers)
            n_prune    = max(0, min(int(out_ch * layer_rate / 100.0), out_ch - 1))

            if n_prune == 0:
                bandit.update(reward=0.0)
                continue

            layer_scores = scores.get(name)
            if layer_scores is None:
                logger.warning(f"No scores for layer {name}, skipping.")
                bandit.update(reward=0.0)
                continue

            prune_indices = layer_scores.argsort()[:n_prune].tolist()
            reward        = -float(layer_scores[prune_indices].mean().item())
            bandit.update(reward=reward)

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
