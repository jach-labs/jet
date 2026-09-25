# Decision Index evaluation

Jet is evaluated with the official [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
runner (`decision_index`, pinned to commit `52a698928a9ae5bdf16b75687c903871db29c6e5`
in the `benchmark` extra). **No official overall Decision Index score has been
measured for any Jet version.** Results so far are sampled or partial diagnostics.

## Engines

| Engine | Model | Backend |
|---|---|---|
| `decision_index_torch:TorchJetEngine` | Jet v6 (Qwen3.5-4B): `michaljach/jet` merged weights, or `Qwen/Qwen3.5-4B` plus an adapter | PyTorch, CUDA |
| `decision_index_engine:JetEngine` | Qwen3-0.6B releases, default `michaljach/jet` revision `25ccbd9e` (V5) | MLX (Metal, or CUDA on Linux) |
| `decision_index_ensemble:TwoOrderJetEngine` | as `JetEngine`, averaging original and reversed option order | MLX |

All engines share one request policy (`decision_index_engine.prepare_request`):
complete prompts or an explicit `Unsupported`, no truncation or option pruning,
exact option keys, and full-precision probabilities, with the chosen option's
probability as confidence. The default complete-prompt limit is 8,192 tokens.
Provenance records the resolved model revision, adapter and calibration hashes,
and code hashes.

`jet-bench-index prepare`, `audit` and `report` build compatibility cases and
complete-group diagnostic samples, audit tokenizer coverage, and report official
native metrics on a diagnostic sample. Diagnostic reports never produce an overall index.

## Running

```sh
# Jet v6 on CUDA
uv sync --extra benchmark --inexact
uv run --no-sync python -m decision_index run --engine decision_index_torch:TorchJetEngine \
  --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
  --out artifacts/decision-index/runs/compatibility-v6 \
  --option model=michaljach/jet --option revision=e5b8f610ddb92ffaba596ae452bed32a9fef49ca \
  --option max_tokens=8192

# Qwen3-0.6B releases with MLX on CUDA (the launcher sets up CUDA headers)
bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
  --out artifacts/decision-index/runs/compatibility-0.6b --option max_tokens=8192
```

Rebuild the public diagnostic corpus, then sample it:

```sh
uv run --extra benchmark python scripts/rebuild_index_diagnostic.py
uv run --extra benchmark jet-bench-index prepare \
  --suite-dir artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt \
  --out artifacts/decision-index/diagnostic --n 500 --allow-partial
```

The official runner resumes from existing results. Keep model revision, capacity
and code fixed within one output directory, and use a new directory per backend.
Run the compatibility file before a full run. Use only the official scorer, and keep
complete source groups, exclusions, full denominators, and track and macro weights.

## Results

- **Jet v6 / Qwen3.5-4B:** 1,900 sampled requests across 15 datasets, with zero
  errors or unsupported requests. The comparison with published entrants is in
  [`experiments/jet-4b-evaluation/results.md`](../experiments/jet-4b-evaluation/results.md).
  The full-benchmark expansion and the remaining blockers to an overall score are in
  [`experiments/jet-4b-full-index/`](../experiments/jet-4b-full-index/README.md).
- **Qwen3-0.6B (jet-1, 2026-09-23):** a partial public rebuild of 23,113 requests
  over eight benchmarks, and a compatibility pass (26/26). Coverage is in
  `docs/bench/decision-index/coverage.json`; `diagnostic-8k.json` is an incomplete
  snapshot of a run that was stopped early.

## Blockers to an overall score

- `multimodalart/decision-index-suite` returns HTTP 404
  ([apolinario/decision-index#1](https://github.com/apolinario/decision-index/issues/1)).
- HLE (`cais/hle`) needs gated-dataset access.
- The public reproduction kit lacks release-v2 recipes for several panel additions.
  Without the matching corpus, any subset average is not comparable to the index.
