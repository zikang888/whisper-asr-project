# Whisper ASR Project - Setup Guide

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

## Installation

### Step 1: Install PyTorch with CUDA support

Choose ONE based on your CUDA version:

```bash
# CUDA 12.1 (recommended for RTX 40xx / H100)
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older cards like RTX 20xx/30xx)
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch>=2.0.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure environment

```bash
cp .env.example .env
# Edit .env to set your paths and preferences
```

### Step 4: Choose Hugging Face endpoint

In `.env`, set `HF_ENDPOINT`:
- China mainland: `HF_ENDPOINT="https://hf-mirror.com"`
- Global: Leave empty `HF_ENDPOINT=""`

### Step 5: Update YAML config paths

Edit files in `configs/` to replace `/root/autodl-tmp/` with your local paths.

## Troubleshooting

- **"CUDA not available"**: You installed CPU-only PyTorch. Reinstall with `--index-url` option.
- **"ImportError: libcudart.so"**: CUDA version mismatch. Check `nvidia-smi` and match PyTorch CUDA version.
- **Failed to download model**: Set `HF_ENDPOINT="https://hf-mirror.com"` in `.env`.
