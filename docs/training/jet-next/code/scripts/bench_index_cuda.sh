#!/usr/bin/env bash
# Decision Index CLI with Jet's MLX CUDA environment. No cloud job or upload.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra cuda --extra benchmark --inexact
export CUDA_HOME="$PWD/.venv/cuda_home"
mkdir -p "$CUDA_HOME/include"
shopt -s nullglob
for pkg in .venv/lib/python*/site-packages/nvidia/{cuda_runtime,cuda_cccl,cuda_nvrtc}/include; do
    for header in "$PWD/$pkg"/*; do
        [ "$(basename "$header")" = __init__.py ] || ln -sfn "$header" "$CUDA_HOME/include/"
    done
done
export MLX_USE_CUDA_GRAPHS=0
export MLX_CUDA_SDPA_CACHE_SIZE="${MLX_CUDA_SDPA_CACHE_SIZE:-2048}"
export MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache"
exec .venv/bin/python -m decision_index "$@"
