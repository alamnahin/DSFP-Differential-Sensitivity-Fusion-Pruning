# DSFP: Differential Sensitivity Fusion Pruning

**Official implementation for:**
> *DSFP: The Differential Sensitivity Fusion Approach to Layerwise DCNN Pruning*
> Iftekhar Haider Chowdhury, Zaed Ikbal Syed, Md. Nahin Alam, Ahmed Faizul Haque Dhrubo*, Mohammad Abdul Qayum
> Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh
> Submitted to *PLOS ONE* — Manuscript PONE-D-26-10141

---

## Overview

DSFP is a structured filter-pruning framework for deep convolutional neural networks. The core idea is to score filter importance not by any single metric, but by the **disagreement** across three independent views of importance. Filters that all three metrics agree are unimportant are pruned with high confidence; filters that only one metric flags are treated with caution. This *stability-through-consensus* criterion is implemented as an exponential fusion of pairwise absolute differences between min-max-normalised scores.

### Three-phase algorithm

**Phase A — Importance scoring** (single forward/backward pass, deterministic):
- `Grad(F)`: L2 norm of the gradient w.r.t. filter weights — captures learning dynamics.
- `Taylor(F)`: First-order Taylor expansion `|w · ∂L/∂w|` — approximates loss sensitivity.
- `KL(F)`: Output-space KL divergence `KL(p_full ‖ p_masked_F)` — measures functional impact on predictions.

All three metrics are min-max normalised to `[0, 1]` per layer before fusion:

```
score(F) = exp(|Grad−Taylor| + |Taylor−KL| + 0.5·|Grad−KL|)
```

The 0.5 coefficient on the third term prevents double-weighting of the gradient signal after normalisation. Higher score = more inter-metric disagreement = prune first.

**Phase B — Layer-wise ratio selection** (contextual bandit, single-step):
A lightweight 2-layer MLP bandit maps `[global_rate, layer_depth]` to a per-layer pruning ratio offset within `±explore_delta` of the global target. One action per layer per pruning call — not sequential Q-learning.

**Phase C — Structural zeroing:**
Selected filters (lowest-score output channels) are zeroed in the Conv2d weights, bias, and the immediately following BatchNorm layer.

Post-pruning recovery uses **knowledge distillation** (KDTrainer): the original model acts as teacher, the pruned model as student, with a dynamically-decaying alpha that transitions from soft-label to hard-label supervision over the fine-tuning budget.

---

## Project Structure

```
DSFP/
├── main.py                         # Entry point: argparse + full pipeline
├── requirements.txt
├── setup.sh                        # One-shot env setup, data download, smoke test
├── run.sh                          # Experiment launcher for all paper experiments
│
├── configs/
│   ├── defaults.py                 # Canonical defaults (used for YAML merge logic)
│   ├── vgg16_cifar10.yaml
│   ├── alexnet_cifar10.yaml
│   ├── resnet56_cifar100.yaml
│   └── resnet18_tiny_imagenet.yaml
│
├── models/
│   └── registry.py        # VGG16CIFAR, AlexNetCIFAR, ResNet56CIFAR, ResNet18-Tiny
│
├── pruning/
│   ├── dsfp.py            # _ImportanceScorer, _BanditAgent, DSFPruner
│   ├── baselines.py       # L1NormPruner, TaylorPruner, SNIPPruner, RandomPruner
│   └── ablation.py        # 12 ablation variants + table printer
│
├── training/
│   └── trainer.py         # BaseTrainer (SGD+Cosine+Mixup), KDTrainer (dynamic-α)
│
└── utils/
    ├── data.py             # CIFAR-10/100/Tiny-ImageNet loaders + calibration split
    ├── metrics.py          # compute_accuracy, compute_flops (thop), filter counts
    └── logging.py          # setup_logging, ResultsLogger → CSV + JSON outputs
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1.0
- CUDA (recommended; CPU works but is slow for KL scoring)

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | Core |
| `torchvision` | ≥ 0.16.0 | Datasets, ResNet-18 backbone |
| `thop` | ≥ 0.1.1 | Correct MFLOPs counting via `profile()` |
| `numpy` | ≥ 1.24.0 | Numerics, calibration split |
| `pyyaml` | ≥ 6.0 | Config file parsing |

---

## Setup

The `setup.sh` script handles everything in one command:

```bash
# Standard (venv, CIFAR-10 + CIFAR-100 only)
bash setup.sh

# Using conda
bash setup.sh --conda

# Also download Tiny-ImageNet (~240 MB)
bash setup.sh --tiny-imagenet

# Both
bash setup.sh --conda --tiny-imagenet
```

What it does:
1. Checks Python ≥ 3.9
2. Creates `.venv/` (or conda env `dsfp`)
3. Installs all dependencies from `requirements.txt`
4. Downloads CIFAR-10 and CIFAR-100 via torchvision
5. Optionally downloads and restructures Tiny-ImageNet val set for `ImageFolder`
6. Runs a 1-epoch smoke test to verify the full stack

---

## Quickstart

### Using a config file (recommended)

```bash
# VGG-16 on CIFAR-10 — full experiment (3 rates × 3 seeds + baselines)
python main.py --config configs/vgg16_cifar10.yaml

# AlexNet on CIFAR-10
python main.py --config configs/alexnet_cifar10.yaml

# ResNet-56 on CIFAR-100
python main.py --config configs/resnet56_cifar100.yaml

# ResNet-18 on Tiny-ImageNet (requires data download first)
python main.py --config configs/resnet18_tiny_imagenet.yaml
```

### Using CLI flags directly

```bash
# Single seed, 70% pruning, DSFP
python main.py \
    --arch vgg16 \
    --dataset cifar10 \
    --pruning-rates 70 \
    --single-seed 42

# Three rates, with all baselines for comparison
python main.py \
    --arch alexnet \
    --dataset cifar10 \
    --pruning-rates 50 60 70 \
    --baselines

# With ablation study (runs 12 variants at 60% pruning)
python main.py \
    --config configs/vgg16_cifar10.yaml \
    --ablation

# Baseline-only run (no DSFP)
python main.py \
    --arch vgg16 \
    --dataset cifar10 \
    --method l1norm \
    --pruning-rates 50 60 70

# Eval only — load a checkpoint and measure accuracy
python main.py \
    --arch vgg16 \
    --dataset cifar10 \
    --pretrained-path results/my_exp/seed42_dsfp_70pct.pth \
    --eval-only \
    --single-seed 42
```

### Using `run.sh` (all paper experiments)

```bash
# Run all paper experiments in sequence
bash run.sh

# Single experiment
bash run.sh --exp vgg16_cifar10
bash run.sh --exp resnet56_cifar100
bash run.sh --exp ablation

# Dry run — print commands without executing
bash run.sh --dry-run

# Specific GPU
bash run.sh --gpu 1

# Single seed for fast debugging
bash run.sh --single-seed 42
```

Available `--exp` values: `vgg16_cifar10` | `alexnet_cifar10` | `resnet56_cifar100` | `resnet18_tiny_imagenet` | `ablation` | `baselines` | `all`

---

## Configuration

All hyperparameters can be set via YAML config file, CLI flags, or both. **CLI flags always override the config file.**

### Full argument reference

| Argument | Default | Description |
|---|---|---|
| `--config` | `None` | Path to YAML config file |
| `--arch` | `vgg16` | Architecture: `vgg16` \| `alexnet` \| `resnet56` \| `resnet18` |
| `--dataset` | `cifar10` | Dataset: `cifar10` \| `cifar100` \| `tiny-imagenet` |
| `--num-classes` | `10` | Number of output classes |
| `--data-dir` | `./data` | Root directory for datasets |
| `--batch-size` | `128` | Training batch size |
| `--calibration-size` | `512` | Samples used for importance scoring |
| `--base-epochs` | `40` | Epochs for base model fine-tuning |
| `--base-lr` | `1e-3` | SGD learning rate |
| `--weight-decay` | `1e-4` | L2 regularisation |
| `--momentum` | `0.9` | SGD momentum |
| `--label-smoothing` | `0.1` | Label smoothing factor |
| `--patience` | `12` | Early stopping patience (base trainer) |
| `--pruning-rates` | `50 60 70` | Target pruning rates (%) — space-separated list |
| `--method` | `dsfp` | Pruning method: `dsfp` \| `l1norm` \| `taylor` \| `snip` \| `random` |
| `--bandit-lr` | `0.01` | Bandit MLP learning rate |
| `--bandit-hidden` | `32` | Bandit MLP hidden size |
| `--bandit-explore-delta` | `5.0` | ±% exploration range around global rate |
| `--bandit-epsilon-decay` | `0.995` | ε-greedy decay factor |
| `--kd-epochs` | `700` | KD fine-tuning epochs |
| `--kd-lr` | `1e-4` | Adam LR for KD fine-tuning |
| `--kd-alpha` | `0.5` | Initial KD loss weight |
| `--kd-temperature` | `4.0` | Softmax temperature T |
| `--kd-patience` | `10` | Early stopping patience (KD trainer) |
| `--kd-dynamic-alpha` | `True` | Linearly decay α from 0.5 to 0.1 |
| `--seeds` | `42 123 2026` | Random seeds for multi-run evaluation |
| `--single-seed` | `None` | Run with one seed only (overrides `--seeds`) |
| `--output-dir` | `./results` | Root output directory |
| `--exp-name` | auto | Experiment subdirectory name |
| `--save-checkpoints` | `False` | Save pruned model `.pth` per seed/rate |
| `--baselines` | `False` | Also run all baseline methods |
| `--ablation` | `False` | Run ablation study (DSFP only) |
| `--skip-base-finetune` | `False` | Use `--pretrained-path` directly |
| `--eval-only` | `False` | Evaluate checkpoint, no training |
| `--amp` | `True` | Automatic mixed precision |
| `--device` | auto | Force `cuda` or `cpu` |

### Example: custom YAML config

```yaml
# configs/my_experiment.yaml
arch: resnet56
dataset: cifar100
num-classes: 100
pruning-rates: [50, 70]
method: dsfp
kd-epochs: 500
seeds: [42]
baselines: false
ablation: true
output-dir: ./results
```

```bash
python main.py --config configs/my_experiment.yaml --single-seed 42
```

---

## Output Files

All results are written to `./results/<exp_name>/`:

| File | Contents |
|---|---|
| `run.log` | Full timestamped training log |
| `results.csv` | One row per (seed × method × pruning_rate) |
| `summary.csv` | Mean ± std across seeds per (method × rate) |
| `results.json` | Full structured dump of all seed results |
| `ablation.csv` | Ablation variant results (if `--ablation`) |
| `seed{N}_{method}_{rate}pct.pth` | Pruned model checkpoints (if `--save-checkpoints`) |

### `summary.csv` columns

```
method, pruning_rate, acc_kd_mean, acc_kd_std, acc_retention_mean,
acc_retention_std, params_M_mean, flops_M_mean, flops_reduction_mean, n_seeds
```

FLOPs are computed with `thop.profile()` and reported in **MFLOPs**. Accuracy retention is `100 × acc_kd / base_acc`.

---

## Baseline Methods

All baselines share the same pruning interface and post-pruning KD fine-tuning schedule as DSFP, making comparisons directly controlled.

| Method | Flag | Description |
|---|---|---|
| L1-Norm | `l1norm` | Prune filters with smallest L1-norm weight (Li et al., ICLR 2017) |
| Taylor | `taylor` | Single-metric `\|w·∂L/∂w\|` only (Molchanov et al., NeurIPS 2019) |
| SNIP | `snip` | Connection sensitivity `\|g·w\|/Σ\|g·w\|` per layer (Lee et al., ICLR 2019) |
| Random | `random` | Uniform random filter selection — lower-bound sanity baseline |

Run all baselines alongside DSFP:

```bash
python main.py --config configs/vgg16_cifar10.yaml --baselines
```

---

## Ablation Study

Run with `--ablation` to evaluate all 12 variants defined in paper Section IV-C. All variants use the same pruning rate (60%), same KD budget, and same random seed for clean attribution.

| Variant | What is removed/changed |
|---|---|
| `grad_only` | Taylor and KL removed |
| `taylor_only` | Grad and KL removed |
| `kl_only` | Grad and Taylor removed |
| `grad_taylor` | KL removed |
| `grad_kl` | Taylor removed |
| `taylor_kl` | Grad removed |
| `dsfp_full` | Full three-metric fusion *(reference)* |
| `arithmetic_sum` | `(g+t+k)/3` instead of exp(diff) |
| `weighted_sum` | `0.4·g + 0.4·t + 0.2·k` instead of exp(diff) |
| `fixed_rate` | Uniform 60% per layer, no bandit |
| `ce_finetune` | Standard cross-entropy fine-tuning, no KD |
| `no_finetune` | No fine-tuning after pruning |

Results are printed as a table and saved to `ablation.csv`.

---

## Supported Architectures and Datasets

| Architecture | Dataset | Classes | Input |
|---|---|---|---|
| VGG-16 (CIFAR-adapted) | CIFAR-10 | 10 | 32×32 |
| VGG-16 (CIFAR-adapted) | CIFAR-100 | 100 | 32×32 |
| AlexNet (CIFAR-adapted) | CIFAR-10 | 10 | 32×32 |
| AlexNet (CIFAR-adapted) | CIFAR-100 | 100 | 32×32 |
| ResNet-56 (He et al., 2016) | CIFAR-10 / CIFAR-100 | 10 / 100 | 32×32 |
| ResNet-18 (Tiny-ImageNet adapted) | Tiny-ImageNet | 200 | 64×64 |

CIFAR-10 and CIFAR-100 download automatically via torchvision. Tiny-ImageNet requires manual download or `bash setup.sh --tiny-imagenet`.

---

## Reproducibility

All experiments use three fixed seeds: **42, 123, 2026**.

Reproducibility is enforced at every stochastic point:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

The calibration subset (512 samples for importance scoring) is drawn from the training set using `np.random.default_rng(seed)` and held fixed across all pruning rates within a seed. All reported accuracy values are **mean ± std across 3 seeds**.

Hardware used for the paper's results: NVIDIA T4 (16 GB VRAM), CUDA 11.8, PyTorch 2.1, Python 3.10.

---

## Known Limitations

- **KL scoring cost:** Computing `KL(F)` requires one masked forward pass per filter per calibration batch. For VGG-16 this takes ~442 ms on a T4 GPU. For wider networks (ResNet-50, >2000 filters) this can be several minutes; batching or approximation via moment matching is a planned extension.
- **MobileNetV2:** Depthwise-separable convolutions require adaptation of the KL masking logic (zeroing a depthwise filter affects the entire channel). Currently not supported.
- **Full ImageNet:** Not evaluated due to GPU resource constraints. ResNet-56/CIFAR-100 and ResNet-18/Tiny-ImageNet serve as the scalability benchmarks in the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@article{chowdhury2026dsfp,
  title   = {{DSFP}: The Differential Sensitivity Fusion Approach to Layerwise {DCNN} Pruning},
  author  = {Chowdhury, Iftekhar Haider and Syed, Zaed Ikbal and Alam, Md. Nahin
             and Dhrubo, Ahmed Faizul Haque and Qayum, Mohammad Abdul},
  journal = {PLOS ONE},
  year    = {2026},
  note    = {Under review — Manuscript PONE-D-26-10141}
}
```

---

## License

This project is released for academic and research use. See `LICENSE` for details.

---

## Contact

Corresponding author: **Ahmed Faizul Haque Dhrubo** — ahmed.dhrubo@northsouth.edu
Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh
