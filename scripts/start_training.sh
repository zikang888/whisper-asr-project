#!/bin/bash

echo "========================================"
echo "Whisper Training - Production Mode"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - Model: whisper-small"
echo "  - Dataset: librispeech_asr (clean)"
echo "  - Batch size: 8 (effective: 32)"
echo "  - Optimizations: FP16, gradient checkpointing"
echo "  - Workers: 4"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Starting training..."
echo "This will run for the configured max_steps (default: 500)"
echo "Press Ctrl+C to stop"
echo ""

python scripts/train_yaml.py --config configs/optimized_config.yaml
