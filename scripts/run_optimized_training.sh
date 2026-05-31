#!/bin/bash

echo "=========================================="
echo "Whisper Model Training - Optimized Version"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/4] Checking environment..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate TroeAI-2

echo "[2/4] Verifying CUDA availability..."
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}'); print(f'  ✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'  ✓ PyTorch: {torch.__version__}')"

echo ""
echo "[3/4] Starting training with optimizations..."
echo "  Config: configs/optimized_config.yaml"
echo "  Script: train_optimized.py"
echo "  Batch size: 8 (effective: 32 with grad accumulation)"
echo "  Workers: 4"
echo "  FP16: enabled"
echo ""

python train_optimized.py --config configs/optimized_config.yaml

echo ""
echo "[4/4] Training completed!"
echo "=========================================="
