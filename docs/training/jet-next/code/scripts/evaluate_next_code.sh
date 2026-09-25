#!/usr/bin/env bash
# Larger fixed CRUXEval check after a regression on the original 55-case sample.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
if [[ -n "${1:-}" ]]; then while kill -0 "$1" 2>/dev/null; do sleep 10; done; fi
[[ -f docs/training/jet-next/isarcasm-en-v4.json ]]
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
    --rows artifacts/decision-index/cruxeval-full-diagnostic.jsonl.gz \
    --out "artifacts/decision-index/runs/cruxeval-full-cuda-next-$name" \
    --option max_tokens=8192 --option max_cached_tokens=8192 "${ARGS[@]}" > "logs/jet-next/$name-cruxeval-full.log" 2>&1
  uv run --no-sync jet-bench-index report --rows artifacts/decision-index/cruxeval-full-diagnostic.jsonl.gz \
    --results "artifacts/decision-index/runs/cruxeval-full-cuda-next-$name/results.jsonl" \
    --out "docs/training/jet-next/cruxeval-full-$name.json" > "logs/jet-next/$name-cruxeval-full-report.log" 2>&1
done
printf 'Full available CRUXEval comparison completed.\n'
