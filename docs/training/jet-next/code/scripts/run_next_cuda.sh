#!/usr/bin/env bash
# Continue after the one-epoch V3 warm-start ablation; all outputs are local.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 CUDA_HOME="$PWD/.venv/cuda_home" MLX_USE_CUDA_GRAPHS=0
export MLX_CUDA_SDPA_CACHE_SIZE=2048 MLX_PTX_CACHE_DIR="$PWD/.venv/ptx_cache"
WARM=adapters/jet-v3-warm-20260924
NEXT=adapters/jet-v4-transfer-20260924
# Waiting PID is supplied only by the live parent session, never discovered by pattern.
if [[ -n "${1:-}" ]]; then
    while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('adapters/jet-v3-warm-20260924')
config=json.loads((p/'training_config.json').read_text())
metrics=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['step']==config['total_steps'], 'Warm-start run did not finish'
PY
bash train_cuda.sh --init-adapter "$WARM" --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
  --train work/jet-next/v4-smoke.jsonl --val work/jet-next/v4-smoke.jsonl \
  --out adapters/jet-v4-transfer-smoke --epochs 1 --lr 1e-5 --max-batch-tokens 3072 \
  --max-state-tokens 2048 --use-example-weights --val-limit 64 --eval-every 10 --seed 230924 \
  > logs/jet-next/v4-smoke.log 2>&1
bash train_cuda.sh --init-adapter "$WARM" --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
  --train data/train_v4_tools_focus.jsonl --val data/selection_v4_tools.jsonl --out "$NEXT" \
  --epochs 1 --rank 16 --lr 1e-5 --grad-checkpoint --max-batch-tokens 3072 \
  --max-state-tokens 2048 --use-example-weights --val-limit 10000 --eval-every 250 --seed 230924 \
  > logs/jet-next/v4-train.log 2>&1
for name in warm v4; do
  if [[ "$name" == warm ]]; then RUN="$WARM"; CAL=data/calibration_v3.jsonl;
  else RUN="$NEXT"; CAL=data/calibration_v4_tools.jsonl; fi
  CONTEXT=4096
  uv run --no-sync jet-calibrate --adapter "$RUN" --data "$CAL" --limit 10000 \
    --batch-size 2 --max-state-tokens "$CONTEXT" > "logs/jet-next/$name-calibration.log" 2>&1
  for dataset in test score_eval test_v3_new test_v4_new test_v4_tools; do
    CONTEXT=4096
    if [[ "$dataset" == test_v4_new || "$dataset" == test_v4_tools ]]; then CONTEXT=2048; fi
    uv run --no-sync jet-eval --adapter "$RUN" --data "data/$dataset.jsonl" --batch-size 2 \
      --max-state-tokens "$CONTEXT" --json "docs/training/jet-next/$name-$dataset.json" \
      > "logs/jet-next/$name-$dataset.log" 2>&1
  done
  bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
    --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
    --out "artifacts/decision-index/runs/diagnostic-cuda-next-$name" \
    --option model=mlx-community/Qwen3-0.6B-bf16 \
    --option revision=42096995f6402fde107068cf530136fe64b604f8 --option adapter="$RUN" \
    --option max_tokens=8192 --option max_cached_tokens=8192 > "logs/jet-next/$name-diagnostic.log" 2>&1
  uv run --no-sync jet-bench-index report --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
    --results "artifacts/decision-index/runs/diagnostic-cuda-next-$name/results.jsonl" \
    --out "docs/training/jet-next/diagnostic-$name.json" > "logs/jet-next/$name-report.log" 2>&1
done
uv run --no-sync jet-eval --base-model models/jet --data data/test_v4_new.jsonl --batch-size 2 \
  --max-state-tokens 2048 --json docs/training/jet-next/baseline-test_v4_new.json \
  > logs/jet-next/baseline-test_v4_new.log 2>&1
uv run --no-sync jet-eval --base-model models/jet --data data/test_v4_tools.jsonl --batch-size 2 \
  --max-state-tokens 2048 --json docs/training/jet-next/baseline-test_v4_tools.json \
  > logs/jet-next/baseline-test_v4_tools.log 2>&1
sha256sum -c work/jet-v3/protected.sha256 > logs/jet-next/protected-check.log
sha256sum -c work/jet-v3/published.sha256 > logs/jet-next/published-check.log
printf 'Next checkpoint training and evaluation completed.\n'
