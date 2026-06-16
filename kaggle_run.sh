#!/usr/bin/env bash
# kaggle_run.sh — run DSFP on Kaggle 2×T4 with auto-resume
# Usage: bash kaggle_run.sh [config_file]
set -euo pipefail

CONFIG="${1:-configs/vgg16_cifar10.yaml}"

echo "=== DSFP Kaggle 2×T4 Runner ==="
echo "Config: $CONFIG"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

# Install missing dependencies
pip install thop --quiet

# Run with resume enabled (default) — safe to interrupt and restart
python main.py \
  --config "$CONFIG" \
  --resume \
  --save-checkpoints \
  --num-workers 2 \
  --output-dir /kaggle/working/results

echo "=== Done. Results in /kaggle/working/results ==="
