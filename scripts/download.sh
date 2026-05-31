#!/bin/bash
# Download Script for Whisper ASR Training Project
# This script downloads the model and dataset

set -e

# Load environment variables
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  export $(cat .env | xargs)
else
  echo "Warning: .env file not found, using default values"
  export HF_HOME="./hf_cache"
  export HF_ENDPOINT="https://hf-mirror.com"
fi

echo "========================================="
echo "  Whisper ASR Training Setup"
echo "========================================="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p ${HF_HOME}
mkdir -p ./models
mkdir -p ./output
mkdir -p ./logs

echo ""
echo "========================================="
echo "  Step 1: Download Pre-trained Model"
echo "========================================="

# Use Python to download the model
python3 << 'EOF'
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/whisper-small')
MODEL_LOCAL_DIR = os.environ.get('MODEL_LOCAL_DIR', './models/whisper-small')

print(f"Downloading model: {MODEL_NAME}")
print(f"To: {MODEL_LOCAL_DIR}")

if os.path.exists(MODEL_LOCAL_DIR):
    print("Model already exists, skipping download.")
else:
    print("Downloading model...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_LOCAL_DIR)
    processor.save_pretrained(MODEL_LOCAL_DIR)
    print("Model saved successfully!")
EOF

echo ""
echo "========================================="
echo "  Step 2: Download Dataset (Optional)"
echo "========================================="
echo "The dataset will be automatically downloaded by the training script"
echo "when you run the training. This step verifies we can access the dataset."
echo ""

python3 << 'EOF'
from datasets import load_dataset

print("Testing dataset access...")
try:
    # Try to load a small subset to verify access
    dataset = load_dataset(
        "librispeech_asr", 
        "clean", 
        split="validation[:10]",
        trust_remote_code=True
    )
    print("✓ Dataset access verified!")
    print(f"  Number of samples: {len(dataset)} samples loaded successfully.")
    print("")
    print("The full dataset will be downloaded when training starts.")
except Exception as e:
    print("⚠️ Dataset download test failed.")
    print(f"Error: {e}")
    print("The dataset will still be downloaded during training.")
EOF

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "What to do next:"
echo "  1. Copy .env.example to .env and edit if needed"
echo "  2. Run training: ./scripts/quick_train.sh (quick test)"
echo "  3. Run full training: ./scripts/full_train.sh"
echo "  4. View README.md for more details"
echo ""
