
#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for DSFP
#
# Designed for Kaggle (2×T4) and standard Linux environments.
# Kaggle uses a managed Python where venv/ensurepip may be broken or absent.
# This script detects that situation and installs directly with pip instead.
#
# What this script does:
#   1. Checks prerequisites (Python >= 3.10)
#   2. Creates a virtual environment IF possible; falls back to --user install
#   3. Installs all Python dependencies from requirements.txt
#   4. Downloads CIFAR-10 and CIFAR-100 via torchvision (auto-download)
#   5. Optionally downloads Tiny-ImageNet
#   6. Runs a smoke test (single-seed, 1 epoch, 1 batch) to verify the stack
#
# Usage:
#   bash setup.sh                   # default: auto-detect env, CIFAR only
#   bash setup.sh --no-venv         # skip venv, always use system pip
#   bash setup.sh --tiny-imagenet   # also download Tiny-ImageNet
#   bash setup.sh --smoke-test      # run smoke test at the end
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
NO_VENV=false
DOWNLOAD_TINY=false
RUN_SMOKE=false
for arg in "$@"; do
    case $arg in
        --no-venv)        NO_VENV=true ;;
        --tiny-imagenet)  DOWNLOAD_TINY=true ;;
        --smoke-test)     RUN_SMOKE=true ;;
        *) warn "Unknown flag: $arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================================"
echo "  DSFP — Environment Setup"
echo "============================================================"
echo ""

# ── 1. Check Python version ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_BIN=$(command -v python3 || command -v python || true)
[ -z "$PYTHON_BIN" ] && error "Python not found. Install Python >= 3.10 and re-run."

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python ${PY_VERSION} detected. DSFP requires Python >= 3.10."
fi
success "Python ${PY_VERSION} found at ${PYTHON_BIN}"

# ── 2. Try venv; fall back to system pip on Kaggle/restricted envs ────────────
VENV_DIR="${SCRIPT_DIR}/.venv"
USING_VENV=false

if [ "$NO_VENV" = false ]; then
    info "Attempting to create venv at ${VENV_DIR}..."
    if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
        # Verify pip inside the venv actually works
        if "${VENV_DIR}/bin/python" -m pip --version &>/dev/null; then
            PYTHON="${VENV_DIR}/bin/python"
            PIP="${VENV_DIR}/bin/pip"
            USING_VENV=true
            success "venv created and pip is functional."
        else
            warn "venv created but pip is broken inside it — falling back to system pip."
            rm -rf "$VENV_DIR"
        fi
    else
        warn "venv creation failed (ensurepip missing — typical on Kaggle/Debian managed Python)."
        warn "Falling back to system pip with --break-system-packages."
    fi
fi

if [ "$USING_VENV" = false ]; then
    # Use system Python + pip, break-system-packages for PEP 668 environments
    PYTHON="$PYTHON_BIN"

    # Determine pip invocation
    if "$PYTHON_BIN" -m pip --version &>/dev/null; then
        PIP="$PYTHON_BIN -m pip"
    else
        error "pip not found. Install pip and re-run (e.g. apt install python3-pip)."
    fi

    # Test if --break-system-packages is needed
    if "$PYTHON_BIN" -m pip install --dry-run pip 2>&1 | grep -q "externally-managed"; then
        PIP="$PYTHON_BIN -m pip --break-system-packages"
        warn "PEP 668 environment detected — using --break-system-packages."
    fi
fi

ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
[ "$USING_VENV" = false ] && ACTIVATE_CMD="# (system Python — no activation needed)"

# ── 3. Upgrade pip & install dependencies ─────────────────────────────────────
info "Upgrading pip..."
$PIP install --quiet --upgrade pip setuptools wheel

info "Installing dependencies from requirements.txt..."
$PIP install --quiet -r "${SCRIPT_DIR}/requirements.txt"
success "Dependencies installed."

# Verify torch is importable
$PYTHON -c "import torch; print(f'    torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" \
    && success "PyTorch import OK." \
    || error "PyTorch import failed after installation."

# ── 4. Download CIFAR-10 / CIFAR-100 ─────────────────────────────────────────
info "Pre-downloading CIFAR-10 and CIFAR-100 (via torchvision)..."
PYTHONPATH="${SCRIPT_DIR}" $PYTHON - <<'PYEOF'
import torchvision.datasets as D, os
data_dir = os.path.join(os.environ.get("SCRIPT_DIR", "."), "data")
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

# ── 5. Optionally download Tiny-ImageNet ──────────────────────────────────────
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

        # Val restructure is handled automatically in utils/data.py on first run
        info "Tiny-ImageNet extracted. Val set will be restructured automatically on first use."
        success "Tiny-ImageNet downloaded at ${TINY_DIR}"
    fi
fi

# ── 6. Smoke test (optional) ──────────────────────────────────────────────────
if [ "$RUN_SMOKE" = true ]; then
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
        --log-level WARNING \
        2>&1 | tail -5

    if [ $? -eq 0 ]; then
        success "Smoke test passed!"
    else
        error "Smoke test failed — check the output above."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "============================================================"
echo ""
if [ "$USING_VENV" = true ]; then
    echo "  Activate environment:"
    echo "    ${ACTIVATE_CMD}"
    echo ""
fi
echo "  Quick-start (Kaggle 2×T4):"
echo "    bash kaggle_run.sh configs/vgg16_cifar10.yaml"
echo ""
echo "  Quick-start (general):"
echo "    python main.py --config configs/vgg16_cifar10.yaml"
echo "    python main.py --arch alexnet --dataset cifar10 --pruning-rates 70 --single-seed 42"
echo "    python main.py --config configs/resnet56_cifar100.yaml"
echo ""
echo "  Results land in: ./results/<exp_name>/"
echo "    results.csv  summary.csv  results.json  run.log"
echo ""
