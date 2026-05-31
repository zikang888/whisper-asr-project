#!/bin/bash
# Full Training Script
# Runs complete training on the full dataset

set -e

echo "========================================="
echo "  Full Training Mode"
echo "========================================="

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# Disable quick mode for full training
export QUICK_TEST_MODE=false

# Run the training
python3 scripts/train.py

echo ""
echo "========================================="
echo "  Full Training Complete!"
echo "========================================="
