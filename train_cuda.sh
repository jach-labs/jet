#!/usr/bin/env bash
# Run jet-train on Linux/CUDA. Extra args go to jet-train.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --extra cuda --inexact --quiet

# MLX JIT-compiles kernels and wants one CUDA_HOME/include; the pip wheels spread the headers
# over several packages, so link them into a single tree.
export CUDA_HOME=$PWD/.venv/cuda_home
if [ ! -d "$CUDA_HOME/include" ]; then
    mkdir -p "$CUDA_HOME/include"
    for pkg in .venv/lib/python*/site-packages/nvidia/{cuda_runtime,cuda_cccl,cuda_nvrtc}/include; do
        for f in "$PWD/$pkg"/*; do
            [ "$(basename "$f")" = __init__.py ] || ln -sfn "$f" "$CUDA_HOME/include/"
        done
    done
fi
export MLX_USE_CUDA_GRAPHS=0                      # batch shapes vary every step, so graphs just thrash
export MLX_PTX_CACHE_DIR=$PWD/.venv/ptx_cache     # keep JIT-compiled kernels across runs
exec uv run --extra cuda --no-sync jet-train "$@"
