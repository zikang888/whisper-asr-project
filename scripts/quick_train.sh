#!/bin/bash
# Quick Training Script
# Runs training on a small subset for verification

set -e

echo "========================================="
echo "  Quick Training (Verification Mode)"
echo "========================================="

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# Set quick mode if not already set
export QUICK_TEST_MODE=true

# Run the training
python3 scripts/train.py

echo ""
echo "========================================="
echo "  Quick Training Complete!"
echo "========================================="
