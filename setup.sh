#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for DSFP
#
# What this script does:
#   1. Checks prerequisites (Python >=3.9, conda or venv)
#   2. Creates a virtual environment (conda or venv)
#   3. Installs all Python dependencies from requirements.txt
#   4. Installs the DSFP package in editable mode
#   5. Downloads CIFAR-10 and CIFAR-100 via torchvision (auto-download)
#   6. Optionally downloads Tiny-ImageNet
#   7. Runs a smoke test (single-seed, 1 epoch, 1 batch) to verify the stack
#
# Usage:
#   bash setup.sh                   # default: venv, CIFAR only
#   bash setup.sh --conda           # use conda instead of venv
#   bash setup.sh --tiny-imagenet   # also download Tiny-ImageNet
#   bash setup.sh --conda --tiny-imagenet
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse flags ───────────────────────────────────────────────────────────────
USE_CONDA=false
DOWNLOAD_TINY=false
for arg in "$@"; do
    case $arg in
        --conda)          USE_CONDA=true  ;;
        --tiny-imagenet)  DOWNLOAD_TINY=true ;;
        *) warn "Unknown flag: $arg" ;;
    esac
done

ENV_NAME="dsfp"
PYTHON_MIN="3.9"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================================"
echo "  DSFP — Environment Setup"
echo "============================================================"
echo ""

# ── 1. Check Python version ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_BIN=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON_BIN" ]; then
    error "Python not found. Install Python >= ${PYTHON_MIN} and re-run."
fi
PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    error "Python ${PY_VERSION} detected. DSFP requires Python >= ${PYTHON_MIN}."
fi
success "Python ${PY_VERSION} found at ${PYTHON_BIN}"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ "$USE_CONDA" = true ]; then
    info "Setting up conda environment '${ENV_NAME}'..."
    if ! command -v conda &>/dev/null; then
        error "conda not found. Install Miniconda/Anaconda or run without --conda."
    fi
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Conda env '${ENV_NAME}' already exists — skipping creation."
    else
        conda create -y -n "${ENV_NAME}" python="${PY_VERSION}" pip
        success "Conda env '${ENV_NAME}' created."
    fi
    ACTIVATE_CMD="conda activate ${ENV_NAME}"
    PIP="conda run -n ${ENV_NAME} pip"
    PYTHON="conda run -n ${ENV_NAME} python"
else
    VENV_DIR="${SCRIPT_DIR}/.venv"
    info "Setting up venv at ${VENV_DIR}..."
    if [ ! -d "$VENV_DIR" ]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        success "venv created at ${VENV_DIR}"
    else
        warn "venv already exists at ${VENV_DIR} — skipping creation."
    fi
    PIP="${VENV_DIR}/bin/pip"
    PYTHON="${VENV_DIR}/bin/python"
    ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
fi

# ── 3. Upgrade pip & install dependencies ─────────────────────────────────────
info "Upgrading pip..."
$PIP install --quiet --upgrade pip setuptools wheel

info "Installing dependencies from requirements.txt..."
$PIP install --quiet -r "${SCRIPT_DIR}/requirements.txt"
success "Dependencies installed."

# ── 4. Install DSFP package (editable) ────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/setup.py" ] || [ -f "${SCRIPT_DIR}/pyproject.toml" ]; then
    info "Installing DSFP in editable mode..."
    $PIP install --quiet -e "${SCRIPT_DIR}"
    success "DSFP installed (editable)."
else
    info "No setup.py / pyproject.toml found — adding project root to PYTHONPATH."
fi

# ── 5. Download CIFAR-10 / CIFAR-100 ─────────────────────────────────────────
info "Pre-downloading CIFAR-10 and CIFAR-100 (via torchvision)..."
$PYTHON - <<'PYEOF'
import torchvision.datasets as D
import os
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)
print("  Downloading CIFAR-10 ...")
D.CIFAR10(data_dir, train=True,  download=True)
D.CIFAR10(data_dir, train=False, download=True)
print("  Downloading CIFAR-100 ...")
D.CIFAR100(data_dir, train=True,  download=True)
D.CIFAR100(data_dir, train=False, download=True)
print("  Done.")
PYEOF
success "CIFAR-10 and CIFAR-100 downloaded to ./data/"

# ── 6. Optionally download Tiny-ImageNet ──────────────────────────────────────
if [ "$DOWNLOAD_TINY" = true ]; then
    TINY_DIR="${SCRIPT_DIR}/data/tiny-imagenet-200"
    if [ -d "${TINY_DIR}" ]; then
        warn "Tiny-ImageNet already present at ${TINY_DIR}."
    else
        info "Downloading Tiny-ImageNet (~240 MB)..."
        mkdir -p "${SCRIPT_DIR}/data"
        TINY_URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        TINY_ZIP="${SCRIPT_DIR}/data/tiny-imagenet-200.zip"

        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "${TINY_ZIP}" "${TINY_URL}"
        elif command -v curl &>/dev/null; then
            curl -L --progress-bar -o "${TINY_ZIP}" "${TINY_URL}"
        else
            error "Neither wget nor curl found. Download manually:\n  ${TINY_URL}\nand extract to ./data/"
        fi

        info "Extracting Tiny-ImageNet..."
        unzip -q "${TINY_ZIP}" -d "${SCRIPT_DIR}/data/"
        rm "${TINY_ZIP}"

        # Fix validation directory structure for ImageFolder compatibility
        info "Restructuring Tiny-ImageNet validation set..."
        $PYTHON - <<'PYEOF'
import os, shutil
val_dir  = os.path.join("data", "tiny-imagenet-200", "val")
ann_file = os.path.join(val_dir, "val_annotations.txt")
if not os.path.exists(ann_file):
    print("  val_annotations.txt not found — skipping restructure.")
    exit()
with open(ann_file) as f:
    lines = f.readlines()
for line in lines:
    parts    = line.strip().split('\t')
    img_file = parts[0]
    class_id = parts[1]
    class_dir = os.path.join(val_dir, "images", class_id)
    os.makedirs(class_dir, exist_ok=True)
    src = os.path.join(val_dir, "images", img_file)
    dst = os.path.join(class_dir, img_file)
    if os.path.exists(src):
        shutil.move(src, dst)
# Rename images/ to flat structure expected by ImageFolder
flat = os.path.join(val_dir, "images")
for cls in os.listdir(flat):
    src = os.path.join(flat, cls)
    dst = os.path.join(val_dir, cls)
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.move(src, dst)
try:
    os.rmdir(flat)
except OSError:
    pass
print("  Validation set restructured.")
PYEOF
        success "Tiny-ImageNet downloaded and prepared at ${TINY_DIR}"
    fi
fi

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
info "Running smoke test (1 seed, 1 base epoch, 1 KD epoch)..."
PYTHONPATH="${SCRIPT_DIR}" $PYTHON "${SCRIPT_DIR}/main.py" \
    --arch vgg16 \
    --dataset cifar10 \
    --data-dir "${SCRIPT_DIR}/data" \
    --base-epochs 1 \
    --kd-epochs 1 \
    --pruning-rates 50 \
    --single-seed 42 \
    --batch-size 64 \
    --calibration-size 64 \
    --skip-base-finetune \
    --method dsfp \
    --output-dir "${SCRIPT_DIR}/results" \
    --exp-name smoke_test \
    --no-save-checkpoints \
    --log-level WARNING \
    2>&1 | tail -5

if [ $? -eq 0 ]; then
    success "Smoke test passed!"
else
    error "Smoke test failed — check the output above."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "============================================================"
echo ""
echo "  Activate environment:"
echo "    ${ACTIVATE_CMD}"
echo ""
echo "  Quick start examples:"
echo "    # VGG-16 on CIFAR-10 (full experiment)"
echo "    python main.py --config configs/vgg16_cifar10.yaml"
echo ""
echo "    # AlexNet, single seed, 70% pruning"
echo "    python main.py --arch alexnet --dataset cifar10 \\"
echo "                   --pruning-rates 70 --single-seed 42"
echo ""
echo "    # ResNet-56 on CIFAR-100"
echo "    python main.py --config configs/resnet56_cifar100.yaml"
echo ""
echo "    # With ablation study"
echo "    python main.py --config configs/vgg16_cifar10.yaml --ablation"
echo ""
echo "    # Baselines only (no DSFP)"
echo "    python main.py --arch vgg16 --dataset cifar10 --method l1norm \\"
echo "                   --pruning-rates 50 60 70"
echo ""
echo "  Results are written to: ./results/<exp_name>/"
echo "    results.csv     — per-run rows"
echo "    summary.csv     — mean ± std across seeds"
echo "    results.json    — full JSON dump"
echo "    run.log         — full training log"
echo ""
