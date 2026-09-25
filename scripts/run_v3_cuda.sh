#!/usr/bin/env bash
# Local-only v3 experiment. Run from repository root after building/auditing data.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export CUDA_HOME="$PWD/.venv/cuda_home"
export MLX_USE_CUDA_GRAPHS=0
export MLX_CUDA_SDPA_CACHE_SIZE=2048
export MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache"
RUN=adapters/jet-v3-decision-20260923
mkdir -p logs/jet-v3 docs/training/jet-v3
if [[ -e "$RUN/adapters.safetensors" ]]; then
    echo "Refusing to replace an existing completed checkpoint: $RUN" >&2; exit 1
fi
bash train_cuda.sh --base-model mlx-community/Qwen3-0.6B-bf16 \
    --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
    --train data/train_v3.jsonl --val data/selection_v3.jsonl --out "$RUN" \
    --epochs 2 --rank 16 --lr 1e-4 --grad-checkpoint --max-batch-tokens 4096 \
    --max-state-tokens 1024 --eval-every 250 --val-limit 10000 --seed 230923 \
    > logs/jet-v3/train.log 2>&1
uv run --no-sync jet-calibrate --adapter "$RUN" --data data/calibration_v3.jsonl \
    --limit 10000 --batch-size 4 > logs/jet-v3/calibration.log 2>&1
BASELINE=$(uv run --no-sync python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("michaljach/jet",revision="8a97cfea2df622bb03f5dc9b02567e21abd2551c"))')
for dataset in test score_eval test_v3_new; do
    uv run --no-sync jet-eval --base-model "$BASELINE" --data "data/$dataset.jsonl" \
        --batch-size 4 --json "docs/training/jet-v3/baseline-$dataset.json" > "logs/jet-v3/baseline-$dataset.log" 2>&1
    uv run --no-sync jet-eval --adapter "$RUN" --data "data/$dataset.jsonl" \
        --batch-size 4 --json "docs/training/jet-v3/v3-$dataset.json" > "logs/jet-v3/v3-$dataset.log" 2>&1
done
bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
    --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
    --out artifacts/decision-index/runs/compatibility-cuda-v3 \
    --option model=mlx-community/Qwen3-0.6B-bf16 \
    --option revision=42096995f6402fde107068cf530136fe64b604f8 \
    --option adapter="$RUN" --option max_tokens=8192 --option max_cached_tokens=8192 \
    > logs/jet-v3/compatibility-v3.log 2>&1
for variant in baseline v3; do
    args=()
    if [[ "$variant" == v3 ]]; then
        args=(--option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 --option adapter="$RUN")
    fi
    bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
        --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
        --out "artifacts/decision-index/runs/diagnostic-cuda-v3-$variant" \
        --option max_tokens=8192 --option max_cached_tokens=8192 "${args[@]}" \
        > "logs/jet-v3/diagnostic-$variant.log" 2>&1
    uv run --no-sync jet-bench-index report --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
        --results "artifacts/decision-index/runs/diagnostic-cuda-v3-$variant/results.jsonl" \
        --out "docs/training/jet-v3/diagnostic-$variant.json" > "logs/jet-v3/report-$variant.log" 2>&1
done
sha256sum -c work/jet-v3/protected.sha256 > logs/jet-v3/protected-check.log
printf 'Training, calibration, and evaluation completed.\n'
