"""
configs/defaults.py — canonical default values, used to detect CLI overrides
when merging a YAML file.
"""

import argparse


def get_default_config() -> argparse.Namespace:
    """Return the canonical default configuration as a Namespace."""
    return argparse.Namespace(
        # Architecture
        arch="vgg16",
        num_classes=10,
        pretrained_path=None,
        # Dataset
        dataset="cifar10",
        data_dir="./data",
        batch_size=128,
        num_workers=2,
        calibration_size=512,
        # Base fine-tuning
        base_epochs=40,
        base_lr=1e-3,
        weight_decay=1e-4,
        momentum=0.9,
        label_smoothing=0.1,
        cosine_t0=50,
        cosine_tmult=2,
        accumulation_steps=4,
        patience=12,
        # Pruning
        pruning_rates=[50.0, 60.0, 70.0],
        method="dsfp",
        # Bandit
        bandit_lr=0.01,
        bandit_buffer_size=512,
        bandit_hidden=32,
        bandit_epsilon_decay=0.995,
        explore_delta=5.0,
        # KD
        kd_epochs=700,
        kd_lr=1e-4,
        kd_weight_decay=1e-4,
        kd_alpha=0.5,
        kd_temperature=4.0,
        kd_patience=10,
        kd_dynamic_alpha=True,
        # Reproducibility
        seeds=[42, 123, 2026],
        single_seed=None,
        # Output
        output_dir="./results",
        exp_name=None,
        save_checkpoints=False,
        log_level="INFO",
        # Modes
        eval_only=False,
        skip_base_finetune=False,
        ablation=False,
        baselines=False,
        # Hardware
        device=None,
        amp=True,
        config=None,
    )
