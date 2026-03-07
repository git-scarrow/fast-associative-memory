#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/dev/projects/fast-associative-memory

# Pull latest
git pull

# Create venv if missing
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install deps (CUDA 12.1 wheels)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Run extraction
python extract_dinov2_vitb14.py

echo "Done. Files saved to ./feature_cache_vitb14/"
