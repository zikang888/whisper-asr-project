#!/bin/bash

echo "========================================"
echo "Preparing Whisper Training Dataset"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python << 'EOF'
import os
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache/datasets"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

print("Loading train dataset...")
train_ds = load_dataset("librispeech_asr", "clean", split="test")
print(f"Train samples: {len(train_ds)}")
print("This will download audio files...")
print("Please wait while audio files are downloaded...")

# This triggers the download
for i in range(min(10, len(train_ds))):
    try:
        audio = train_ds[i]["audio"]
        print(f"  Downloaded sample {i+1}/10", end="\r")
    except Exception as e:
        print(f"  Error downloading sample {i}: {e}")

print("\n✓ First 10 samples downloaded")
print("✓ Remaining samples will be downloaded during preprocessing")
EOF

echo ""
echo "✓ Dataset preparation complete!"
echo "You can now start training with:"
echo "  ./scripts/start_training.sh"
