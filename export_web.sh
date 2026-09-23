#!/usr/bin/env bash
# Build the browser model for onnxruntime-web from a fused model dir (see src/onnx_web.py):
#   ./export_web.sh models/jet        -> models/jet/onnx/model_q8.onnx, checked against models/jet/golden.json
# Needs Linux, an NVIDIA GPU and systemd. The export loads the weights on the GPU, and every step runs under
# a memory cap, so running out of memory kills that step instead of the desktop. A CPU export of the 0.6B model
# peaks near 6 GB and took down a 16 GB machine. The tools live in their own environment (torch, optimum),
# outside the project's.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-models/jet}
BUILD=${JET_WEB_BUILD:-$HOME/.cache/jet-web-build}
ENV=$BUILD/env
capped() { systemd-run --user --scope --quiet -p MemoryMax="${JET_WEB_MEMORY:-5G}" -p MemorySwapMax=0 "$@"; }

if [ ! -x "$ENV/bin/python" ]; then
    uv venv --quiet --python 3.12 "$ENV"
    uv pip install --quiet --python "$ENV/bin/python" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --quiet --python "$ENV/bin/python" optimum==2.1.0 "optimum-onnx[onnxruntime]==0.1.0" transformers==4.57.6 \
        onnx==1.23.0 onnx_ir==1.0.0 onnxruntime==1.30.0 accelerate==1.15.0
fi
[ -f "$MODEL/golden.json" ] || { echo "no $MODEL/golden.json: run jet-golden --base-model $MODEL --out $MODEL/golden.json first" >&2; exit 1; }

rm -rf "$BUILD/export"
capped "$ENV/bin/optimum-cli" export onnx --model "$MODEL" --task text-generation-with-past --device cuda \
    --no-post-process "$BUILD/export"
capped "$ENV/bin/python" src/onnx_web.py slice "$BUILD/export/model.onnx" "$BUILD/export/model_sliced.onnx"
capped "$ENV/bin/python" src/onnx_web.py quantize "$BUILD/export/model_sliced.onnx" "$MODEL/onnx/model_q8.onnx"
capped "$ENV/bin/python" src/onnx_web.py check "$MODEL/onnx/model_q8.onnx" "$MODEL/golden.json"
