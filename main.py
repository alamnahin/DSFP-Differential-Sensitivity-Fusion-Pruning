"""
DSFP: Differential Sensitivity Fusion Pruning
Main entry point — with atomic checkpointing for Kaggle 2×T4 sessions.

Checkpointing logic
-------------------
A manifest file tracks which (seed, method, rate, stage) have completed.
On restart, completed stages are skipped and in-progress stages resume
from the last saved model checkpoint.  All writes use os.replace() so
a disconnection mid-write never corrupts checkpoints.

Multi-GPU (2×T4)
----------------
When two CUDA devices are visible, the script wraps the model in
nn.DataParallel for base fine-tuning and KD fine-tuning, then unwraps
before pruning (pruning operates on the raw module).

Usage:
    python main.py --config configs/vgg16_cifar10.yaml
    python main.py --arch vgg16 --dataset cifar10 --pruning-rates 50 60 70
    python main.py --arch resnet18 --dataset tiny-imagenet --single-seed 42

Fixes applied vs original:
  [BUG-22] get_model() now requires dataset kwarg (registry.py BUG-12 fix).
           All call sites updated.
  [BUG-23] compute_flops() now accepts dataset kwarg for correct input size.
  [BUG-24] get_dataloaders() now passes device for pin_memory guard.
  [BUG-25] VGG16CIFAR has 5 MaxPool layers → output is 1×1 after features,
           so the linear dim is 512*1*1=512. This is correct but only because
           of AdaptiveAvgPool-less architecture; documented.
"""

import argparse
import logging
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml

from configs.defaults import get_default_config
from training.trainer import BaseTrainer, KDTrainer
from pruning.dsfp import DSFPruner
from pruning.baselines import get_baseline_pruner
from utils.data import get_dataloaders
from utils.metrics import compute_accuracy, compute_flops, count_parameters, count_nonzero_filters
from utils.logging import setup_logging, ResultsLogger
from utils.checkpoint import AtomicCheckpointer
from models.registry import get_model, load_model


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSFP: Differential Sensitivity Fusion Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arch", type=str, default="vgg16",
                        choices=["vgg16", "alexnet", "resnet56", "resnet18"])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tiny-imagenet"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--base-epochs", type=int, default=40)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cosine-t0", type=int, default=50)
    parser.add_argument("--cosine-tmult", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--pruning-rates", type=float, nargs="+",
                        default=[50, 60, 70])
    parser.add_argument("--method", type=str, default="dsfp",
                        choices=["dsfp", "l1norm", "taylor", "snip", "random"])
    parser.add_argument("--bandit-lr", type=float, default=0.01)
    parser.add_argument("--bandit-buffer-size", type=int, default=512)
    parser.add_argument("--bandit-hidden", type=int, default=32)
    parser.add_argument("--bandit-epsilon-decay", type=float, default=0.995)
    parser.add_argument("--bandit-explore-delta", type=float, default=5.0)
    parser.add_argument("--kd-epochs", type=int, default=700)
    parser.add_argument("--kd-lr", type=float, default=1e-4)
    parser.add_argument("--kd-weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--kd-patience", type=int, default=10)
    parser.add_argument("--kd-dynamic-alpha", action="store_true", default=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-base-finetune", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    # Kaggle-specific
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoints if available (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    defaults = vars(get_default_config())
    cli_overrides = {k for k, v in vars(args).items()
                     if v != defaults.get(k)}
    for key, value in cfg.items():
        key_norm = key.replace("-", "_")
        if key_norm not in cli_overrides:
            setattr(args, key_norm, value)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# Multi-GPU helpers
# ===========================================================================

def _wrap_dp(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap in DataParallel if multiple GPUs are available."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger_ = logging.getLogger(__name__)
        logger_.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel to get the raw module."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ===========================================================================
# Per-seed pipeline
# ===========================================================================

def run_single_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    logger: logging.Logger,
    results_logger: ResultsLogger,
    ckpt: AtomicCheckpointer,
) -> dict:
    """
    Full pipeline for one seed: base fine-tune → prune → KD fine-tune.
    Skips stages already completed (atomic checkpoint manifest).
    """
    set_seed(seed)
    logger.info(f"{'='*60}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"{'='*60}")

    # ── Try to load a previously completed result ──────────────────────────
    if args.resume and ckpt.is_done("seed_complete", seed=seed):
        cached = ckpt.load_result("seed_result", seed=seed)
        if cached is not None:
            logger.info(f"[Resume] Seed {seed} already complete — loading cached result")
            return cached

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, calib_loader = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calibration_size=args.calibration_size,
        seed=seed,
        device=device,   # [BUG-24]
    )

    # ── Base model ────────────────────────────────────────────────────────────
    if args.pretrained_path:
        base_model = load_model(args.arch, args.pretrained_path,
                                num_classes=args.num_classes,
                                dataset=args.dataset,       # [BUG-22]
                                device=device)
        logger.info(f"Loaded pretrained weights from {args.pretrained_path}")
    else:
        base_model = get_model(args.arch, num_classes=args.num_classes,
                               dataset=args.dataset,        # [BUG-22]
                               device=device)

    # ── Base fine-tuning (with checkpoint resume) ──────────────────────────
    base_acc = None
    if not args.skip_base_finetune:
        if args.resume and ckpt.checkpoint_exists("base", seed=seed):
            logger.info(f"[Resume] Loading base model checkpoint for seed {seed}")
            base_model, extra = ckpt.load_model(base_model, "base",
                                                seed=seed, device=device)
            base_acc = extra.get("best_acc")
            if base_acc is None:
                base_acc = compute_accuracy(base_model, test_loader, device)
        else:
            logger.info("Fine-tuning base model...")
            base_model_dp = _wrap_dp(base_model, device)
            base_trainer  = BaseTrainer(
                model=base_model_dp,
                device=device,
                lr=args.base_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                cosine_t0=args.cosine_t0,
                cosine_tmult=args.cosine_tmult,
                accumulation_steps=args.accumulation_steps,
                patience=args.patience,
                use_amp=args.amp,
                use_mixup=True,
            )
            _, base_acc = base_trainer.train(train_loader, test_loader,
                                             epochs=args.base_epochs)
            base_model  = _unwrap(base_model_dp)
            # Atomic save
            ckpt.save_model(base_model, "base", seed=seed,
                            extra={"best_acc": base_acc})
            ckpt.mark_done("base_finetune", seed=seed)
            logger.info(f"Base model accuracy: {base_acc:.2f}%")
    else:
        if args.resume and ckpt.checkpoint_exists("base", seed=seed):
            base_model, extra = ckpt.load_model(base_model, "base",
                                                seed=seed, device=device)
            base_acc = extra.get("best_acc")
        if base_acc is None:
            base_acc = compute_accuracy(base_model, test_loader, device)
        logger.info(f"Base model accuracy (no fine-tune): {base_acc:.2f}%")

    base_params  = count_parameters(base_model)
    base_flops   = compute_flops(base_model, dataset=args.dataset, device=device)  # [BUG-23]
    base_filters = count_nonzero_filters(base_model)

    logger.info(f"Base — Params: {base_params/1e6:.4f}M | "
                f"FLOPs: {base_flops:.2f} MFLOPs | Filters: {base_filters}")

    seed_results = {
        "seed": seed,
        "base_acc": base_acc,
        "base_params_M": round(base_params / 1e6, 4),
        "base_flops_M": round(base_flops, 2),
        "base_filters": base_filters,
        "pruning_results": [],
    }

    # ── Pruning loop ──────────────────────────────────────────────────────────
    methods = [args.method]
    if args.baselines:
        methods += [m for m in ["l1norm", "taylor", "snip", "random"]
                    if m != args.method]

    for method in methods:
        for rate in args.pruning_rates:
            stage_key = f"prune_kd:{method}:{int(rate)}"
            logger.info(f"\n--- Method: {method.upper()} | Rate: {rate}% ---")

            # Resume: skip already-completed (method, rate) combinations
            if args.resume and ckpt.is_done(stage_key, seed=seed):
                cached = ckpt.load_result(stage_key, seed=seed)
                if cached is not None:
                    logger.info(f"[Resume] Skipping {stage_key} (already done)")
                    seed_results["pruning_results"].append(cached)
                    results_logger.log_entry(seed=seed, **cached)
                    continue

            # Fresh model copy per (method, rate)
            if args.pretrained_path:
                pruned_model = load_model(args.arch, args.pretrained_path,
                                          num_classes=args.num_classes,
                                          dataset=args.dataset,
                                          device=device)
            else:
                pruned_model = deepcopy(base_model)

            # Resume: load pruned-but-not-KD model if available
            kd_start_model_tag = f"post_prune:{method}:{int(rate)}"
            if (args.resume and
                    ckpt.is_done(f"pruned:{method}:{int(rate)}", seed=seed) and
                    ckpt.checkpoint_exists(kd_start_model_tag, seed=seed,
                                           method=method, rate=rate)):
                logger.info(f"[Resume] Loading post-prune model for KD fine-tune")
                pruned_model, _ = ckpt.load_model(pruned_model,
                                                   kd_start_model_tag,
                                                   seed=seed, method=method,
                                                   rate=rate, device=device)
                acc_pruned = compute_accuracy(pruned_model, test_loader, device)
            else:
                # Build pruner and prune
                if method == "dsfp":
                    pruner = DSFPruner(
                        bandit_lr=args.bandit_lr,
                        bandit_buffer_size=args.bandit_buffer_size,
                        bandit_hidden=args.bandit_hidden,
                        bandit_epsilon_decay=args.bandit_epsilon_decay,
                        explore_delta=args.bandit_explore_delta,
                        device=device,
                    )
                else:
                    pruner = get_baseline_pruner(method, device=device)

                pruned_model = pruner.prune(
                    model=pruned_model,
                    dataloader=calib_loader,
                    pruning_rate=rate,
                )
                acc_pruned = compute_accuracy(pruned_model, test_loader, device)
                logger.info(f"Accuracy after pruning (before KD): {acc_pruned:.2f}%")

                # Save pruned model (before KD) for potential restart
                ckpt.save_model(pruned_model, kd_start_model_tag,
                                seed=seed, method=method, rate=rate,
                                extra={"acc_pruned": acc_pruned})
                ckpt.mark_done(f"pruned:{method}:{int(rate)}", seed=seed)

            # KD fine-tuning
            pruned_model_dp = _wrap_dp(pruned_model, device)
            kd_trainer = KDTrainer(
                teacher_model=deepcopy(base_model),
                student_model=pruned_model_dp,
                device=device,
                lr=args.kd_lr,
                weight_decay=args.kd_weight_decay,
                alpha=args.kd_alpha,
                temperature=args.kd_temperature,
                patience=args.kd_patience,
                dynamic_alpha=args.kd_dynamic_alpha,
                use_amp=args.amp,
            )
            acc_kd = kd_trainer.train(train_loader, test_loader,
                                      epochs=args.kd_epochs)
            pruned_model = _unwrap(pruned_model_dp)
            logger.info(f"Accuracy after KD fine-tuning: {acc_kd:.2f}%")

            params_pruned  = count_parameters(pruned_model)
            flops_pruned   = compute_flops(pruned_model, dataset=args.dataset,   # [BUG-23]
                                           device=device)
            filters_pruned = count_nonzero_filters(pruned_model)
            retention      = 100.0 * acc_kd / base_acc

            logger.info(
                f"  Params: {params_pruned/1e6:.4f}M "
                f"({100*(1-params_pruned/base_params):.1f}% reduction) | "
                f"FLOPs: {flops_pruned:.2f} MFLOPs "
                f"({100*(1-flops_pruned/base_flops):.1f}% reduction) | "
                f"Acc retention: {retention:.2f}%"
            )

            entry = {
                "method": method,
                "pruning_rate": rate,
                "acc_pruned": round(acc_pruned, 4),
                "acc_kd": round(acc_kd, 4),
                "acc_retention_pct": round(retention, 2),
                "params_M": round(params_pruned / 1e6, 4),
                "params_reduction_pct": round(100*(1-params_pruned/base_params), 2),
                "flops_M": round(flops_pruned, 2),
                "flops_reduction_pct": round(100*(1-flops_pruned/base_flops), 2),
                "filters": filters_pruned,
            }
            seed_results["pruning_results"].append(entry)
            results_logger.log_entry(seed=seed, **entry)

            # Atomic checkpoint: final model + result
            ckpt.save_model(pruned_model, f"final:{method}:{int(rate)}",
                            seed=seed, method=method, rate=rate,
                            extra=entry)
            ckpt.save_result(entry, stage_key, seed=seed)
            ckpt.mark_done(stage_key, seed=seed)

    # ── Ablation study ────────────────────────────────────────────────────────
    if args.ablation and args.method == "dsfp":
        if args.resume and ckpt.is_done("ablation", seed=seed):
            logger.info(f"[Resume] Skipping ablation for seed {seed}")
        else:
            from pruning.ablation import run_ablation
            ablation_results = run_ablation(
                base_model=base_model,
                base_acc=base_acc,
                train_loader=train_loader,
                test_loader=test_loader,
                calib_loader=calib_loader,
                pruning_rate=60.0,
                kd_epochs=args.kd_epochs,
                kd_lr=args.kd_lr,
                kd_alpha=args.kd_alpha,
                kd_temperature=args.kd_temperature,
                device=device,
                args=args,
            )
            seed_results["ablation"] = ablation_results
            results_logger.log_ablation(seed=seed, results=ablation_results)
            ckpt.mark_done("ablation", seed=seed)

    # Mark entire seed done and cache the result
    ckpt.save_result(seed_results, "seed_result", seed=seed)
    ckpt.mark_done("seed_complete", seed=seed)
    return seed_results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    args = merge_config(args)

    if args.exp_name is None:
        rates_str    = "_".join(str(int(r)) for r in args.pruning_rates)
        args.exp_name = f"{args.arch}_{args.dataset}_{args.method}_r{rates_str}"

    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)

    log_path = os.path.join(args.output_dir, args.exp_name, "run.log")
    logger   = setup_logging(log_path, level=args.log_level)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Args: {vars(args)}")

    # ── Device (2×T4 aware) ──────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        n_gpu = torch.cuda.device_count()
        logger.info(f"Device: {device} | GPUs: {n_gpu}")
        for i in range(n_gpu):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info(f"Device: {device}")

    # ── Checkpointer ─────────────────────────────────────────────────────────
    ckpt = AtomicCheckpointer(args.output_dir, args.exp_name)

    seeds = [args.single_seed] if args.single_seed is not None else args.seeds
    results_logger = ResultsLogger(
        output_dir=os.path.join(args.output_dir, args.exp_name)
    )
    all_results = []

    for seed in seeds:
        result = run_single_seed(args, seed, device, logger, results_logger, ckpt)
        all_results.append(result)

    results_logger.summarize(all_results)
    logger.info("All seeds complete. Summary written to results directory.")


if __name__ == "__main__":
    main()
