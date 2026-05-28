"""
DSFP: Differential Sensitivity Fusion Pruning
Main entry point for training, pruning, and evaluation.

Usage:
    python main.py --config configs/vgg16_cifar10.yaml
    python main.py --arch vgg16 --dataset cifar10 --pruning-rates 50 60 70
    python main.py --arch alexnet --dataset cifar10 --pruning-rates 70 --kd-epochs 700
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml

from configs.defaults import get_default_config
from training.trainer import BaseTrainer, KDTrainer
from pruning.dsfp import DSFPruner
from pruning.baselines import get_baseline_pruner
from utils.data import get_dataloaders
from utils.metrics import compute_accuracy, compute_flops, count_parameters, count_nonzero_filters
from utils.logging import setup_logging, ResultsLogger
from models.registry import get_model, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSFP: Differential Sensitivity Fusion Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file (overrides all defaults; CLI args override config file) ──
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # ── Architecture ──
    parser.add_argument("--arch", type=str, default="vgg16",
                        choices=["vgg16", "alexnet", "resnet56", "resnet18"],
                        help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained weights (.pth)")

    # ── Dataset ──
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tiny-imagenet"],
                        help="Dataset name")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-size", type=int, default=512,
                        help="Number of calibration samples for importance scoring")

    # ── Base model fine-tuning ──
    parser.add_argument("--base-epochs", type=int, default=40,
                        help="Epochs to fine-tune base model before pruning")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cosine-t0", type=int, default=50)
    parser.add_argument("--cosine-tmult", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12,
                        help="Early stopping patience (epochs)")

    # ── Pruning ──
    parser.add_argument("--pruning-rates", type=float, nargs="+", default=[50, 60, 70],
                        help="List of target pruning rates (%%)")
    parser.add_argument("--method", type=str, default="dsfp",
                        choices=["dsfp", "l1norm", "taylor", "snip", "random"],
                        help="Pruning importance method")

    # ── Bandit agent ──
    parser.add_argument("--bandit-lr", type=float, default=0.01)
    parser.add_argument("--bandit-buffer-size", type=int, default=512)
    parser.add_argument("--bandit-hidden", type=int, default=32)
    parser.add_argument("--bandit-epsilon-decay", type=float, default=0.995)
    parser.add_argument("--bandit-explore-delta", type=float, default=5.0,
                        help="±%% exploration range around base pruning rate")

    # ── KD fine-tuning ──
    parser.add_argument("--kd-epochs", type=int, default=700,
                        help="Epochs for knowledge distillation fine-tuning")
    parser.add_argument("--kd-lr", type=float, default=1e-4)
    parser.add_argument("--kd-weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=0.5,
                        help="Weight for KD loss (1-alpha for CE loss)")
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--kd-patience", type=int, default=10,
                        help="Early stopping patience for KD fine-tuning")
    parser.add_argument("--kd-dynamic-alpha", action="store_true", default=True,
                        help="Decay alpha linearly over KD epochs")

    # ── Reproducibility ──
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026],
                        help="Random seeds for multi-run evaluation")
    parser.add_argument("--single-seed", type=int, default=None,
                        help="Run with a single seed (overrides --seeds)")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--save-checkpoints", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])

    # ── Modes ──
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate a given checkpoint, no training")
    parser.add_argument("--skip-base-finetune", action="store_true",
                        help="Skip base model fine-tuning (use pretrained-path directly)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study over DSFP components")
    parser.add_argument("--baselines", action="store_true",
                        help="Also run baseline pruning methods for comparison")

    # ── Hardware ──
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or None (auto-detect)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")

    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config file into args (CLI args take precedence)."""
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


def run_single_seed(args: argparse.Namespace, seed: int,
                    device: torch.device, logger: logging.Logger,
                    results_logger: ResultsLogger) -> dict:
    """Full pipeline for one seed: base fine-tune → prune → KD fine-tune."""
    set_seed(seed)
    logger.info(f"{'='*60}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, calib_loader = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calibration_size=args.calibration_size,
        seed=seed,
    )

    # ── Base model ────────────────────────────────────────────────────────────
    if args.pretrained_path:
        base_model = load_model(args.arch, args.pretrained_path,
                                num_classes=args.num_classes, device=device)
        logger.info(f"Loaded pretrained weights from {args.pretrained_path}")
    else:
        base_model = get_model(args.arch, num_classes=args.num_classes,
                               device=device)

    # ── Base fine-tuning ──────────────────────────────────────────────────────
    if not args.skip_base_finetune:
        logger.info("Fine-tuning base model...")
        base_trainer = BaseTrainer(
            model=base_model,
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
        base_model, base_acc = base_trainer.train(
            train_loader, test_loader, epochs=args.base_epochs
        )
        logger.info(f"Base model accuracy: {base_acc:.2f}%")
    else:
        base_acc = compute_accuracy(base_model, test_loader, device)
        logger.info(f"Base model accuracy (no fine-tune): {base_acc:.2f}%")

    base_params   = count_parameters(base_model)
    base_flops    = compute_flops(base_model, device=device)
    base_filters  = count_nonzero_filters(base_model)

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
            logger.info(f"\n--- Method: {method.upper()} | Rate: {rate}% ---")

            # Fresh model copy per rate
            if args.pretrained_path:
                pruned_model = load_model(args.arch, args.pretrained_path,
                                          num_classes=args.num_classes,
                                          device=device)
            else:
                from copy import deepcopy
                pruned_model = deepcopy(base_model)

            # Pruner
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

            # KD fine-tuning
            kd_trainer = KDTrainer(
                teacher_model=base_model,
                student_model=pruned_model,
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
            logger.info(f"Accuracy after KD fine-tuning: {acc_kd:.2f}%")

            params_pruned  = count_parameters(pruned_model)
            flops_pruned   = compute_flops(pruned_model, device=device)
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

            if args.save_checkpoints:
                ckpt_path = os.path.join(
                    args.output_dir, args.exp_name,
                    f"seed{seed}_{method}_{int(rate)}pct.pth"
                )
                torch.save(pruned_model.state_dict(), ckpt_path)

    # ── Ablation study ────────────────────────────────────────────────────────
    if args.ablation and args.method == "dsfp":
        from pruning.ablation import run_ablation
        ablation_results = run_ablation(
            base_model=base_model,
            base_acc=base_acc,
            train_loader=train_loader,
            test_loader=test_loader,
            calib_loader=calib_loader,
            pruning_rate=60.0,        # fixed ablation rate per paper
            kd_epochs=args.kd_epochs,
            kd_lr=args.kd_lr,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            device=device,
            args=args,
        )
        seed_results["ablation"] = ablation_results
        results_logger.log_ablation(seed=seed, results=ablation_results)

    return seed_results


def main() -> None:
    args = parse_args()
    args = merge_config(args)

    # ── Experiment name ───────────────────────────────────────────────────────
    if args.exp_name is None:
        rates_str = "_".join(str(int(r)) for r in args.pruning_rates)
        args.exp_name = f"{args.arch}_{args.dataset}_{args.method}_r{rates_str}"

    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, args.exp_name, "run.log")
    logger   = setup_logging(log_path, level=args.log_level)
    logger.info(f"Experiment: {args.exp_name}")
    logger.info(f"Args: {vars(args)}")

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Seeds ─────────────────────────────────────────────────────────────────
    seeds = [args.single_seed] if args.single_seed is not None else args.seeds

    results_logger = ResultsLogger(
        output_dir=os.path.join(args.output_dir, args.exp_name)
    )
    all_results = []

    for seed in seeds:
        result = run_single_seed(args, seed, device, logger, results_logger)
        all_results.append(result)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    results_logger.summarize(all_results)
    logger.info("All seeds complete. Summary written to results directory.")


if __name__ == "__main__":
    main()
