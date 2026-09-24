#!/usr/bin/env bash
# Take a fresh rented Linux GPU box to a working setup.
# Usage: MLFLOW_TRACKING_URI=http://host:5000 HF_TOKEN=... bash scripts/bootstrap.sh
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
nvidia-smi -L
uv run python -c "import torch, bitsandbytes; print('cuda', torch.cuda.is_available(), 'bnb', bitsandbytes.__version__)"
if [ -n "${HF_TOKEN:-}" ]; then uv run huggingface-cli login --token "$HF_TOKEN"; fi
echo "Ready. Smoke test: uv run python -m lab.train --config configs/sft_qlora.yaml --set data.max_samples=64 --set train.max_steps=5"
