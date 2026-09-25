#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_HOME="$PWD/.venv/cuda_home" MLX_USE_CUDA_GRAPHS=0
export MLX_CUDA_SDPA_CACHE_SIZE=2048 MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache"
if [[ -n "${1:-}" ]]; then while kill -0 "$1" 2>/dev/null; do sleep 10; done; fi
[[ -f docs/training/jet-next/baseline-test_v4_tools.json ]]
for dataset in test score_eval test_v3_new; do
  uv run --no-sync jet-eval --base-model models/jet --data "data/$dataset.jsonl" \
    --batch-size 2 --max-state-tokens 4096 --json "docs/training/jet-next/baseline-$dataset.json" \
    > "logs/jet-next/baseline-$dataset.log" 2>&1
done
for dataset in test score_eval test_v3_new test_v4_new test_v4_tools; do
  CONTEXT=4096
  if [[ "$dataset" == test_v4_new || "$dataset" == test_v4_tools ]]; then CONTEXT=2048; fi
  uv run --no-sync jet-eval --adapter adapters/jet-v3-decision-20260923 --data "data/$dataset.jsonl" \
    --batch-size 2 --max-state-tokens "$CONTEXT" --json "docs/training/jet-next/v3-$dataset.json" \
    > "logs/jet-next/v3-$dataset.log" 2>&1
done
for name in baseline v3 warm v4; do
  if [[ "$name" == baseline ]]; then
    ARGS=(--option model=michaljach/jet --option revision=8a97cfea2df622bb03f5dc9b02567e21abd2551c)
  else
    case "$name" in
      v3) RUN=adapters/jet-v3-decision-20260923;;
      warm) RUN=adapters/jet-v3-warm-20260924;;
      v4) RUN=adapters/jet-v4-transfer-20260924;;
    esac
    ARGS=(--option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 --option adapter="$RUN")
  fi
  bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
    --rows artifacts/decision-index/esci-diagnostic/diagnostic.jsonl.gz \
    --out "artifacts/decision-index/runs/esci-cuda-next-$name" \
    --option max_tokens=8192 --option max_cached_tokens=8192 "${ARGS[@]}" > "logs/jet-next/$name-esci.log" 2>&1
  uv run --no-sync jet-bench-index report --rows artifacts/decision-index/esci-diagnostic/diagnostic.jsonl.gz \
    --results "artifacts/decision-index/runs/esci-cuda-next-$name/results.jsonl" \
    --out "docs/training/jet-next/esci-$name.json" > "logs/jet-next/$name-esci-report.log" 2>&1
  bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
    --rows artifacts/decision-index/isarcasm-english-diagnostic.jsonl.gz \
    --out "artifacts/decision-index/runs/isarcasm-en-cuda-next-$name" \
    --option max_tokens=8192 --option max_cached_tokens=8192 "${ARGS[@]}" > "logs/jet-next/$name-isarcasm-en.log" 2>&1
  uv run --no-sync jet-bench-index report --rows artifacts/decision-index/isarcasm-english-diagnostic.jsonl.gz \
    --results "artifacts/decision-index/runs/isarcasm-en-cuda-next-$name/results.jsonl" \
    --out "docs/training/jet-next/isarcasm-en-$name.json" > "logs/jet-next/$name-isarcasm-en-report.log" 2>&1
done
printf 'Additional heldouts and ESCI diagnostics completed.\n'
