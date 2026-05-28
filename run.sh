#!/usr/bin/env bash
# =============================================================================
# run.sh — Parameterised experiment launcher for DSFP
#
# Runs all experiments reported in the paper, or a selected subset.
# Results land in ./results/<exp_name>/ as CSV + JSON + log.
#
# Usage:
#   bash run.sh                        # run all paper experiments
#   bash run.sh --exp vgg16_cifar10    # single experiment
#   bash run.sh --exp ablation         # ablation study only
#   bash run.sh --dry-run              # print commands, don't execute
#   bash run.sh --gpu 0                # use specific GPU
#   bash run.sh --single-seed 42       # one seed only (fast CI/debug)
#
# Supported --exp values:
#   vgg16_cifar10 | alexnet_cifar10 | resnet56_cifar100
#   resnet18_tiny_imagenet | ablation | baselines | all (default)
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[run.sh]${NC} $*"; }
success() { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${NC} $*"; }
error()   { echo -e "${RED}[run.sh]${NC} $*"; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="all"
DRY_RUN=false
GPU=""
SINGLE_SEED_FLAG=""
EXTRA_FLAGS=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)          EXP="$2";                       shift 2 ;;
        --dry-run)      DRY_RUN=true;                   shift   ;;
        --gpu)          GPU="$2";                       shift 2 ;;
        --single-seed)  SINGLE_SEED_FLAG="--single-seed $2"; shift 2 ;;
        --extra)        EXTRA_FLAGS="$2";               shift 2 ;;
        *) warn "Unknown argument: $1"; shift ;;
    esac
done

# ── Resolve Python & PYTHONPATH ───────────────────────────────────────────────
VENV="${SCRIPT_DIR}/.venv"
if [ -f "${VENV}/bin/python" ]; then
    PYTHON="${VENV}/bin/python"
elif command -v conda &>/dev/null && conda env list | grep -q "^dsfp "; then
    PYTHON="conda run -n dsfp python"
else
    PYTHON=$(command -v python3 || command -v python)
    warn "No venv or conda env found — using system Python: ${PYTHON}"
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ── GPU setup ─────────────────────────────────────────────────────────────────
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    info "Using GPU: ${GPU}"
fi

# ── Timing helper ─────────────────────────────────────────────────────────────
_elapsed() {
    local secs=$1
    printf "%dh %02dm %02ds" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# ── Command runner ────────────────────────────────────────────────────────────
run_cmd() {
    local desc="$1"; shift
    local cmd="$*"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Starting: ${desc}"
    echo "  CMD: ${cmd}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$DRY_RUN" = true ]; then
        warn "[DRY RUN] Would execute: ${cmd}"
        return
    fi

    local start=$SECONDS
    if eval "$cmd"; then
        local elapsed=$(( SECONDS - start ))
        success "Completed: ${desc} in $(_elapsed $elapsed)"
    else
        error "FAILED: ${desc}"
    fi
}

# ── Experiment definitions ────────────────────────────────────────────────────

exp_vgg16_cifar10() {
    run_cmd "VGG-16 / CIFAR-10 — DSFP (50%, 60%, 70%)" \
        "$PYTHON ${SCRIPT_DIR}/main.py \
            --config ${SCRIPT_DIR}/configs/vgg16_cifar10.yaml \
            --baselines \
            ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
}

exp_alexnet_cifar10() {
    run_cmd "AlexNet / CIFAR-10 — DSFP (50%, 60%, 70%)" \
        "$PYTHON ${SCRIPT_DIR}/main.py \
            --config ${SCRIPT_DIR}/configs/alexnet_cifar10.yaml \
            --baselines \
            ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
}

exp_resnet56_cifar100() {
    run_cmd "ResNet-56 / CIFAR-100 — DSFP (50%, 70%)" \
        "$PYTHON ${SCRIPT_DIR}/main.py \
            --config ${SCRIPT_DIR}/configs/resnet56_cifar100.yaml \
            --baselines \
            ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
}

exp_resnet18_tiny_imagenet() {
    TINY_DIR="${SCRIPT_DIR}/data/tiny-imagenet-200"
    if [ ! -d "$TINY_DIR" ]; then
        warn "Tiny-ImageNet not found at ${TINY_DIR}."
        warn "Run: bash setup.sh --tiny-imagenet  to download it first."
        return
    fi
    run_cmd "ResNet-18 / Tiny-ImageNet — DSFP (50%, 70%)" \
        "$PYTHON ${SCRIPT_DIR}/main.py \
            --config ${SCRIPT_DIR}/configs/resnet18_tiny_imagenet.yaml \
            --baselines \
            ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
}

exp_ablation() {
    run_cmd "Ablation Study — VGG-16 / CIFAR-10 at 60%" \
        "$PYTHON ${SCRIPT_DIR}/main.py \
            --config ${SCRIPT_DIR}/configs/vgg16_cifar10.yaml \
            --ablation \
            --pruning-rates 60 \
            --exp-name vgg16_cifar10_dsfp_ablation \
            ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
}

exp_baselines() {
    for method in l1norm taylor snip random; do
        run_cmd "VGG-16 / CIFAR-10 — ${method} baseline" \
            "$PYTHON ${SCRIPT_DIR}/main.py \
                --arch vgg16 \
                --dataset cifar10 \
                --method ${method} \
                --pruning-rates 50 60 70 \
                --exp-name vgg16_cifar10_${method} \
                ${SINGLE_SEED_FLAG} ${EXTRA_FLAGS}"
    done
}

# ── Main dispatch ─────────────────────────────────────────────────────────────
OVERALL_START=$SECONDS

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        DSFP — Experiment Runner                          ║"
echo "║  Experiment set : ${EXP}"
echo "║  Dry run        : ${DRY_RUN}"
echo "║  Single seed    : ${SINGLE_SEED_FLAG:-all seeds}"
echo "╚══════════════════════════════════════════════════════════╝"

case "$EXP" in
    vgg16_cifar10)          exp_vgg16_cifar10 ;;
    alexnet_cifar10)        exp_alexnet_cifar10 ;;
    resnet56_cifar100)      exp_resnet56_cifar100 ;;
    resnet18_tiny_imagenet) exp_resnet18_tiny_imagenet ;;
    ablation)               exp_ablation ;;
    baselines)              exp_baselines ;;
    all)
        exp_vgg16_cifar10
        exp_alexnet_cifar10
        exp_resnet56_cifar100
        exp_resnet18_tiny_imagenet
        exp_ablation
        ;;
    *)
        error "Unknown --exp value: '${EXP}'. "
              "Choose: vgg16_cifar10 | alexnet_cifar10 | resnet56_cifar100 | "
              "resnet18_tiny_imagenet | ablation | baselines | all"
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - OVERALL_START ))
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo -e "║  ${GREEN}All experiments finished${NC} in $(_elapsed $ELAPSED)"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Results directory: ./results/"
echo "║"

if [ -d "${SCRIPT_DIR}/results" ] && [ "$DRY_RUN" = false ]; then
    for d in "${SCRIPT_DIR}/results"/*/; do
        name=$(basename "$d")
        summary="${d}summary.csv"
        if [ -f "$summary" ]; then
            n_rows=$(( $(wc -l < "$summary") - 1 ))
            echo "║    ${name}"
            echo "║      └─ summary.csv  (${n_rows} rows)"
        fi
    done
fi

echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Convenience: print best DSFP result if results exist ─────────────────────
if [ "$DRY_RUN" = false ]; then
    $PYTHON - <<'PYEOF' 2>/dev/null || true
import os, csv, glob

result_dirs = glob.glob(os.path.join("results", "*dsfp*"))
if not result_dirs:
    exit()

print("\n  DSFP Best Results (across all completed experiments):")
print(f"  {'Experiment':<35} {'Method':<8} {'Rate%':>6} "
      f"{'Acc_KD_mean':>12} {'Acc_KD_std':>11} {'FLOP↓%':>8}")
print("  " + "-" * 84)

for d in sorted(result_dirs):
    summary = os.path.join(d, "summary.csv")
    if not os.path.exists(summary):
        continue
    exp_name = os.path.basename(d)
    with open(summary) as f:
        for row in csv.DictReader(f):
            if row.get("method", "") == "dsfp":
                print(
                    f"  {exp_name:<35} {row['method']:<8} "
                    f"{float(row['pruning_rate']):>6.0f} "
                    f"{float(row['acc_kd_mean']):>12.4f} "
                    f"{float(row['acc_kd_std']):>11.4f} "
                    f"{float(row['flops_reduction_mean']):>8.2f}"
                )
print()
PYEOF
fi
