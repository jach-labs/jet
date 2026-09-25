#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_HOME="$PWD/.venv/cuda_home" MLX_USE_CUDA_GRAPHS=0
export MLX_CUDA_SDPA_CACHE_SIZE=2048 MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache"
NEXT=adapters/jet-v5-panel02-r2-20260924
BASE=adapters/jet-v4-transfer-20260924
bash train_cuda.sh --init-adapter "$BASE" --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
 --train work/jet-v5/smoke.jsonl --val work/jet-v5/smoke.jsonl --out adapters/jet-v5-panel02-r2-smoke \
 --epochs 1 --lr 1e-5 --rank 16 --grad-checkpoint --max-batch-tokens 3072 --max-state-tokens 2048 \
 --eval-every 10 --val-limit 100 --seed 240925 > logs/jet-v5/smoke.log 2>&1
bash train_cuda.sh --init-adapter "$BASE" --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
 --train data/train_v5_r2.jsonl --val data/selection_v5.jsonl --out "$NEXT" \
 --epochs 1 --lr 1e-5 --rank 16 --grad-checkpoint --max-batch-tokens 3072 --max-state-tokens 2048 \
 --use-example-weights --eval-every 250 --val-limit 10000 --seed 240925 > logs/jet-v5/train.log 2>&1
.venv/bin/jet-calibrate --adapter "$NEXT" --data data/calibration_v5.jsonl --limit 10000 --batch-size 2 \
 --max-state-tokens 4096 > logs/jet-v5/calibration.log 2>&1
for name in v4 v5; do
 if [[ "$name" == v4 ]]; then RUN="$BASE"; else RUN="$NEXT"; fi
 for dataset in test_v5_new test test_v4_new test_v4_tools; do
  CONTEXT=2048; if [[ "$dataset" == test ]]; then CONTEXT=4096; fi
  .venv/bin/jet-eval --adapter "$RUN" --data "data/$dataset.jsonl" --batch-size 2 \
   --max-state-tokens "$CONTEXT" --json "docs/training/jet-v5/$name-$dataset.json" > "logs/jet-v5/$name-$dataset.log" 2>&1
 done
 bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz --out "artifacts/decision-index/runs/diagnostic-v5-$name" \
  --option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 \
  --option adapter="$RUN" --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/$name-diagnostic.log" 2>&1
 .venv/bin/jet-bench-index report --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
  --results "artifacts/decision-index/runs/diagnostic-v5-$name/results.jsonl" --out "docs/training/jet-v5/$name-diagnostic.json" > "logs/jet-v5/$name-report.log" 2>&1
done
sha256sum -c work/jet-v3/protected.sha256 > logs/jet-v5/protected.log
sha256sum -c work/jet-v3/published.sha256 > logs/jet-v5/published.log
printf 'V5 training and evaluation completed.\n'
