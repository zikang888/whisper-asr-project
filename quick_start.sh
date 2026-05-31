#!/bin/bash

echo "========================================"
echo "Whisper Training - Quick Start Script"
echo "========================================"
echo ""

cd /root/autodl-tmp/asr-whisper

echo "[1/3] Pre-downloading and caching dataset..."
python -c "
from datasets import load_dataset, Audio
print('  Loading train dataset...')
train_ds = load_dataset('librispeech_asr', 'clean', split='test')
print(f'  Train samples: {len(train_ds)}')
print('  Casting audio column...')
train_ds = train_ds.cast_column('audio', Audio(sampling_rate=16000))
print('  Downloading audio files...')
# Just download first 100 samples for quick start
for i in range(min(100, len(train_ds))):
    _ = train_ds[i]['audio']
    if i % 20 == 0:
        print(f'  Downloaded {i}/{min(100, len(train_ds))} samples...')
print('  ✓ Train dataset cached')

print()
print('  Loading eval dataset...')
eval_ds = load_dataset('librispeech_asr', 'clean', split='test')
eval_ds = eval_ds.select(range(min(50, len(eval_ds))))
eval_ds = eval_ds.cast_column('audio', Audio(sampling_rate=16000))
print('  Downloading eval audio files...')
for i in range(len(eval_ds)):
    _ = eval_ds[i]['audio']
    if i % 10 == 0:
        print(f'  Downloaded {i}/{len(eval_ds)} samples...')
print('  ✓ Eval dataset cached')
print()
print('✓ Dataset preparation complete!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "[2/3] Dataset preparation successful!"
    echo ""
    echo "[3/3] Starting training..."
    echo "  Config: configs/quick_test_config.yaml"
    echo "  This will run a quick 50-step training to verify everything works"
    echo ""
    python train.py --config configs/quick_test_config.yaml
else
    echo ""
    echo "✗ Dataset preparation failed!"
    exit 1
fi
