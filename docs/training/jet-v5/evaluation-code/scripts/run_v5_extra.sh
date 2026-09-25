#!/usr/bin/env bash
# Run only after run_v5_cuda.sh finishes; uses the GPU sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_HOME="$PWD/.venv/cuda_home" MLX_USE_CUDA_GRAPHS=0 MLX_CUDA_SDPA_CACHE_SIZE=2048
export MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache" PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_v5_order.py --adapter adapters/jet-v5-panel02-r2-20260924 > logs/jet-v5/inference-selection.log 2>&1
ENGINE=decision_index_engine:JetEngine
if .venv/bin/python -c 'import json,sys;sys.exit(json.load(open("docs/training/jet-v5/inference-selection.json"))["selected"]!="two_order")'; then
 ENGINE=decision_index_ensemble:TwoOrderJetEngine
fi
for SUITE in diagnostic v5-diagnostic-1400; do
 .venv/bin/python -m decision_index run --engine "$ENGINE" \
  --rows "artifacts/decision-index/$SUITE/compatibility.jsonl.gz" \
  --out "artifacts/decision-index/runs/v5-compatibility-$SUITE" \
  --option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 \
  --option adapter=adapters/jet-v5-panel02-r2-20260924 --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/compatibility-$SUITE.log" 2>&1
done
 .venv/bin/jet-eval --base-model models/jet --data data/test_v5_new.jsonl --batch-size 2 \
  --max-state-tokens 2048 --json docs/training/jet-v5/baseline-test_v5_new.json > logs/jet-v5/baseline-test_v5_new.log 2>&1
for NAME in baseline v4 v5; do
 ADAPTER=adapters/jet-v4-transfer-20260924
 if [[ "$NAME" == v5 ]]; then ADAPTER=adapters/jet-v5-panel02-r2-20260924; fi
 OPTIONS=(--option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 --option adapter="$ADAPTER")
 if [[ "$NAME" == baseline ]]; then
  OPTIONS=(--option model=michaljach/jet --option revision=8a97cfea2df622bb03f5dc9b02567e21abd2551c)
 fi
 .venv/bin/python -m decision_index run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/v5-diagnostic-1400/diagnostic.jsonl.gz \
  --out "artifacts/decision-index/runs/v5-expanded-$NAME" \
  "${OPTIONS[@]}" --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/$NAME-expanded.log" 2>&1
 .venv/bin/jet-bench-index report --rows artifacts/decision-index/v5-diagnostic-1400/diagnostic.jsonl.gz \
  --results "artifacts/decision-index/runs/v5-expanded-$NAME/results.jsonl" --out "docs/training/jet-v5/$NAME-expanded.json" > "logs/jet-v5/$NAME-expanded-report.log" 2>&1
done
if .venv/bin/python -c 'import json,sys;sys.exit(json.load(open("docs/training/jet-v5/inference-selection.json"))["selected"]!="two_order")'; then
 for PAIR in 'diagnostic/diagnostic.jsonl.gz v5-two-order-original' 'v5-diagnostic-1400/diagnostic.jsonl.gz v5-two-order-expanded'; do
  read -r ROWS RUN <<< "$PAIR"
  .venv/bin/python -m decision_index run --engine decision_index_ensemble:TwoOrderJetEngine \
   --rows "artifacts/decision-index/$ROWS" --out "artifacts/decision-index/runs/$RUN" \
   --option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 \
   --option adapter=adapters/jet-v5-panel02-r2-20260924 --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/$RUN.log" 2>&1
  .venv/bin/jet-bench-index report --rows "artifacts/decision-index/$ROWS" \
   --results "artifacts/decision-index/runs/$RUN/results.jsonl" --out "docs/training/jet-v5/$RUN.json" > "logs/jet-v5/$RUN-report.log" 2>&1
 done
fi
for MODE in single selected; do
 TARGET=decision_index_engine:JetEngine
 if [[ "$MODE" == selected ]]; then
  [[ "$ENGINE" != decision_index_engine:JetEngine ]] || continue
  TARGET="$ENGINE"
 fi
 .venv/bin/python -m decision_index run --engine "$TARGET" \
  --rows artifacts/decision-index/cruxeval-full-diagnostic.jsonl.gz \
  --out "artifacts/decision-index/runs/v5-code-$MODE" \
  --option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 \
  --option adapter=adapters/jet-v5-panel02-r2-20260924 --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/code-$MODE.log" 2>&1
 .venv/bin/jet-bench-index report --rows artifacts/decision-index/cruxeval-full-diagnostic.jsonl.gz \
  --results "artifacts/decision-index/runs/v5-code-$MODE/results.jsonl" --out "docs/training/jet-v5/code-$MODE.json" > "logs/jet-v5/code-$MODE-report.log" 2>&1
done
for ENTRY in 'esci-diagnostic/diagnostic.jsonl.gz esci' 'isarcasm-english-diagnostic.jsonl.gz sarcasm'; do
 read -r ROWS LABEL <<< "$ENTRY"
 for MODE in single selected; do
  TARGET=decision_index_engine:JetEngine
  if [[ "$MODE" == selected ]]; then
   [[ "$ENGINE" != decision_index_engine:JetEngine ]] || continue
   TARGET="$ENGINE"
  fi
  .venv/bin/python -m decision_index run --engine "$TARGET" \
   --rows "artifacts/decision-index/$ROWS" --out "artifacts/decision-index/runs/v5-$LABEL-$MODE" \
   --option model=mlx-community/Qwen3-0.6B-bf16 --option revision=42096995f6402fde107068cf530136fe64b604f8 \
   --option adapter=adapters/jet-v5-panel02-r2-20260924 --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-v5/$LABEL-$MODE.log" 2>&1
  .venv/bin/jet-bench-index report --rows "artifacts/decision-index/$ROWS" \
   --results "artifacts/decision-index/runs/v5-$LABEL-$MODE/results.jsonl" --out "docs/training/jet-v5/$LABEL-$MODE.json" > "logs/jet-v5/$LABEL-$MODE-report.log" 2>&1
 done
done
printf 'V5 expanded evaluation complete.\n'
