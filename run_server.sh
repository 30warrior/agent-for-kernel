#!/usr/bin/env bash
set -euo pipefail
ASTRA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ASTRA_PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$ASTRA_PROJECT_DIR/.cache"
exec "${ASTRA_PYTHON:-$ASTRA_PROJECT_DIR/.venv/bin/python}" -B -u kernel_optimizer.py "$@"
