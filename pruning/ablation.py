"""
pruning/ablation.py
Systematic ablation study for DSFP components (Section IV-C of the paper).

Ablation variants evaluated at a fixed pruning rate (60% per paper):
  ── Metric ablations (remove one metric at a time) ──
  1.  grad_only        : Grad(F) only, no Taylor, no KL
  2.  taylor_only      : Taylor(F) only
  3.  kl_only          : KL(F) only
  4.  grad_taylor      : Grad + Taylor (no KL)
  5.  grad_kl          : Grad + KL (no Taylor)
  6.  taylor_kl        : Taylor + KL (no Grad)
  7.  dsfp_full        : Full three-metric DSFP fusion (reference)

  ── Fusion form ablations ──
  8.  arithmetic_sum   : (g + t + k) / 3  instead of exp(diff)
  9.  weighted_sum     : 0.4·g + 0.4·t + 0.2·k  (heuristic weights)

  ── Ratio selector ablations ──
  10. fixed_rate       : Uniform 60% per layer (no bandit)

  ── Fine-tuning ablations ──
  11. ce_finetune      : Standard cross-entropy fine-tuning (no KD)
  12. no_finetune      : No fine-tuning at all after pruning

Each variant runs with the same bandit (where applicable), same KD budget,
and same random state so differences are attributable to the component only.
"""

from __future__ import annotations
import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pruning.dsfp import (
    _ImportanceScorer, _BanditAgent, _get_conv_layers,
    _zero_filters, _minmax_norm,
)
from utils.metrics import compute_accuracy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variant importance score builders
# ---------------------------------------------------------------------------

def _scores_grad_only(scorer: _ImportanceScorer, model: nn.Module,
                      calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    out = {}
    for name, conv in _get_conv_layers(model):
        g = scorer._grad_sensitivity(conv).detach()
        out[name] = _minmax_norm(g)
    model.zero_grad()
    return out


def _scores_taylor_only(scorer: _ImportanceScorer, model: nn.Module,
                        calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    out = {}
    for name, conv in _get_conv_layers(model):
        t = scorer._taylor_expansion(conv).detach()
        out[name] = _minmax_norm(t)
    model.zero_grad()
    return out


def _scores_kl_only(scorer: _ImportanceScorer, model: nn.Module,
                    calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    images, _ = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    out = {}
    for name, conv in _get_conv_layers(model):
        k = scorer._kl_divergence(model, name, conv, images).detach()
        out[name] = _minmax_norm(k)
    return out


def _scores_grad_taylor(scorer: _ImportanceScorer, model: nn.Module,
                        calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    out = {}
    for name, conv in _get_conv_layers(model):
        g = _minmax_norm(scorer._grad_sensitivity(conv).detach())
        t = _minmax_norm(scorer._taylor_expansion(conv).detach())
        out[name] = torch.exp(torch.abs(g - t))
    model.zero_grad()
    return out


def _scores_grad_kl(scorer: _ImportanceScorer, model: nn.Module,
                    calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    grad_raw = {n: scorer._grad_sensitivity(c).detach()
                for n, c in _get_conv_layers(model)}
    model.zero_grad()
    out = {}
    for name, conv in _get_conv_layers(model):
        g = _minmax_norm(grad_raw[name])
        k = _minmax_norm(scorer._kl_divergence(model, name, conv, images).detach())
        out[name] = torch.exp(torch.abs(g - k))
    return out


def _scores_taylor_kl(scorer: _ImportanceScorer, model: nn.Module,
                      calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    taylor_raw = {n: scorer._taylor_expansion(c).detach()
                  for n, c in _get_conv_layers(model)}
    model.zero_grad()
    out = {}
    for name, conv in _get_conv_layers(model):
        t = _minmax_norm(taylor_raw[name])
        k = _minmax_norm(scorer._kl_divergence(model, name, conv, images).detach())
        out[name] = torch.exp(torch.abs(t - k))
    return out


def _scores_arithmetic_sum(scorer: _ImportanceScorer, model: nn.Module,
                            calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    """Ablation: plain arithmetic mean of three normalised metrics."""
    full_scores = scorer.compute(model, calib_loader)
    # Re-derive g, t, k from the full scorer to get raw normalised metrics
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    grad_raw   = {n: _minmax_norm(scorer._grad_sensitivity(c).detach())
                  for n, c in _get_conv_layers(model)}
    taylor_raw = {n: _minmax_norm(scorer._taylor_expansion(c).detach())
                  for n, c in _get_conv_layers(model)}
    model.zero_grad()
    out = {}
    for name, conv in _get_conv_layers(model):
        g = grad_raw[name]
        t = taylor_raw[name]
        k = _minmax_norm(scorer._kl_divergence(model, name, conv, images).detach())
        out[name] = (g + t + k) / 3.0
    return out


def _scores_weighted_sum(scorer: _ImportanceScorer, model: nn.Module,
                         calib_loader: DataLoader) -> dict[str, torch.Tensor]:
    """Ablation: weighted sum 0.4·g + 0.4·t + 0.2·k."""
    model.train()
    images, labels = next(iter(calib_loader))
    images = images.to(scorer.device, non_blocking=True)
    labels = labels.to(scorer.device, non_blocking=True)
    model.zero_grad()
    F.cross_entropy(model(images), labels).backward()
    grad_raw   = {n: _minmax_norm(scorer._grad_sensitivity(c).detach())
                  for n, c in _get_conv_layers(model)}
    taylor_raw = {n: _minmax_norm(scorer._taylor_expansion(c).detach())
                  for n, c in _get_conv_layers(model)}
    model.zero_grad()
    out = {}
    for name, conv in _get_conv_layers(model):
        g = grad_raw[name]
        t = taylor_raw[name]
        k = _minmax_norm(scorer._kl_divergence(model, name, conv, images).detach())
        out[name] = 0.4 * g + 0.4 * t + 0.2 * k
    return out


# ---------------------------------------------------------------------------
# Pruning with a given score dict or fixed-rate
# ---------------------------------------------------------------------------

def _apply_scores(model: nn.Module, scores: dict[str, torch.Tensor],
                  pruning_rate: float,
                  bandit_cfg: Optional[dict] = None) -> nn.Module:
    """Apply scores with optional bandit rate selector."""
    conv_layers = _get_conv_layers(model)
    n_layers    = len(conv_layers)
    if bandit_cfg is not None:
        bandit = _BanditAgent(**bandit_cfg)
    else:
        bandit = None

    for layer_idx, (name, conv) in enumerate(conv_layers):
        out_ch = conv.out_channels
        if bandit is not None:
            rate = bandit.select_rate(pruning_rate, layer_idx, n_layers)
        else:
            rate = pruning_rate  # fixed uniform rate

        n_prune = max(0, min(int(out_ch * rate / 100.0), out_ch - 1))
        if n_prune == 0:
            if bandit:
                bandit.update(0.0)
            continue

        layer_scores  = scores.get(name)
        if layer_scores is None:
            if bandit:
                bandit.update(0.0)
            continue

        prune_indices = layer_scores.argsort()[:n_prune].tolist()
        if bandit:
            bandit.update(-float(layer_scores[prune_indices].mean().item()))
        _zero_filters(model, name, prune_indices)
    return model


# ---------------------------------------------------------------------------
# KD and CE fine-tune helpers (thin wrappers to avoid circular import)
# ---------------------------------------------------------------------------

def _kd_finetune(student: nn.Module, teacher: nn.Module,
                 train_loader: DataLoader, test_loader: DataLoader,
                 kd_epochs: int, kd_lr: float, kd_alpha: float,
                 kd_temperature: float, device: torch.device) -> float:
    from training.trainer import KDTrainer
    trainer = KDTrainer(
        teacher_model=teacher,
        student_model=student,
        device=device,
        lr=kd_lr,
        weight_decay=1e-4,
        alpha=kd_alpha,
        temperature=kd_temperature,
        patience=10,
        dynamic_alpha=True,
        use_amp=False,
    )
    return trainer.train(train_loader, test_loader, epochs=kd_epochs)


def _ce_finetune(model: nn.Module,
                 train_loader: DataLoader, test_loader: DataLoader,
                 kd_epochs: int, kd_lr: float,
                 device: torch.device) -> float:
    from training.trainer import BaseTrainer
    trainer = BaseTrainer(
        model=model,
        device=device,
        lr=kd_lr,
        momentum=0.9,
        weight_decay=1e-4,
        label_smoothing=0.0,
        cosine_t0=50,
        cosine_tmult=2,
        accumulation_steps=1,
        patience=10,
        use_amp=False,
        use_mixup=False,
    )
    _, acc = trainer.train(train_loader, test_loader, epochs=kd_epochs)
    return acc


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    base_model: nn.Module,
    base_acc: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    calib_loader: DataLoader,
    pruning_rate: float,
    kd_epochs: int,
    kd_lr: float,
    kd_alpha: float,
    kd_temperature: float,
    device: torch.device,
    args: Any,
) -> list[dict]:
    """
    Run all ablation variants and return a list of result dicts.

    Each dict contains: variant, acc_kd, acc_retention_pct, params_M, flops_M.
    """
    from utils.metrics import compute_flops, count_parameters

    scorer     = _ImportanceScorer(device=device)
    bandit_cfg = dict(
        lr=args.bandit_lr,
        buffer_size=args.bandit_buffer_size,
        hidden=args.bandit_hidden,
        epsilon_decay=args.bandit_epsilon_decay,
        explore_delta=args.bandit_explore_delta,
        device=device,
    )

    # Pre-compute full DSFP scores (shared across metric-ablation variants)
    logger.info("Ablation: computing full DSFP scores once...")
    full_scores = scorer.compute(deepcopy(base_model), calib_loader)

    variants = [
        # (variant_name, score_fn_or_scores, use_bandit, finetune_mode)
        ("grad_only",      _scores_grad_only,       True,  "kd"),
        ("taylor_only",    _scores_taylor_only,     True,  "kd"),
        ("kl_only",        _scores_kl_only,         True,  "kd"),
        ("grad_taylor",    _scores_grad_taylor,      True,  "kd"),
        ("grad_kl",        _scores_grad_kl,          True,  "kd"),
        ("taylor_kl",      _scores_taylor_kl,        True,  "kd"),
        ("dsfp_full",      full_scores,             True,  "kd"),
        ("arithmetic_sum", _scores_arithmetic_sum,  True,  "kd"),
        ("weighted_sum",   _scores_weighted_sum,    True,  "kd"),
        ("fixed_rate",     full_scores,             False, "kd"),
        ("ce_finetune",    full_scores,             True,  "ce"),
        ("no_finetune",    full_scores,             True,  "none"),
    ]

    results = []
    for variant, score_src, use_bandit, ft_mode in variants:
        logger.info(f"  Ablation variant: {variant}")
        model_copy = deepcopy(base_model).to(device)

        # Build or reuse scores
        if callable(score_src):
            scores = score_src(scorer, deepcopy(base_model).to(device),
                               calib_loader)
        else:
            scores = score_src   # pre-computed dict

        # Prune
        _apply_scores(
            model_copy, scores, pruning_rate,
            bandit_cfg=bandit_cfg if use_bandit else None,
        )

        # Fine-tune
        if ft_mode == "kd":
            acc = _kd_finetune(model_copy, base_model,
                               train_loader, test_loader,
                               kd_epochs, kd_lr, kd_alpha,
                               kd_temperature, device)
        elif ft_mode == "ce":
            acc = _ce_finetune(model_copy, train_loader, test_loader,
                               kd_epochs, kd_lr, device)
        else:  # no_finetune
            acc = compute_accuracy(model_copy, test_loader, device)

        params_m  = count_parameters(model_copy) / 1e6
        flops_m   = compute_flops(model_copy, device=device)
        retention = 100.0 * acc / base_acc

        logger.info(f"    {variant}: acc={acc:.4f}% retention={retention:.2f}%")
        results.append({
            "variant":            variant,
            "acc_kd":             round(acc, 4),
            "acc_retention_pct":  round(retention, 2),
            "params_M":           round(params_m, 4),
            "flops_M":            round(flops_m, 2),
        })

    _print_ablation_table(results, base_acc)
    return results


def _print_ablation_table(results: list[dict], base_acc: float) -> None:
    header = f"{'Variant':<20} {'Acc_KD':>9} {'Retention%':>11} {'Params_M':>10} {'FLOPs_M':>9}"
    print("\n" + "=" * len(header))
    print(f"Ablation Study (base acc = {base_acc:.4f}%)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        marker = " ◄" if r["variant"] == "dsfp_full" else ""
        print(f"{r['variant']:<20} {r['acc_kd']:>9.4f} "
              f"{r['acc_retention_pct']:>11.2f} "
              f"{r['params_M']:>10.4f} {r['flops_M']:>9.2f}{marker}")
    print("=" * len(header) + "\n")
