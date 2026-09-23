# Decision Index baseline and Arch handoff

Updated 2026-09-23. Prepared for the `codex/decision-index-training` branch.
Existing deployment, inference and Kev benchmark changes are included so the
Arch checkout has the complete current project.

## What is implemented

- `src/decision_index_engine.py`: `decision_index_engine:JetEngine`, an official
  engine adapter using MLX (Metal locally; CUDA on Linux). Uses the published
  Jet checkpoint `8a97cfea2df622bb03f5dc9b02567e21abd2551c`, its native prompt,
  label-token readout, published choice temperature, and shared-prefix cache.
- Complete prompts or explicit `Unsupported`; no truncation or option pruning.
  The application encoder still retains its existing truncation behavior; this
  adapter deliberately bypasses it. No serving API behavior was changed.
- Default complete-prompt limit 8,192 tokens, configurable up to the checkpoint's
  declared 40,960. A configured capacity is not evidence of long-context quality.
- Full-precision probabilities, exact option keys, chosen-option probability as
  benchmark confidence. Provenance records resolved model revision and code hashes.
- `jet-bench-index prepare`, `audit`, and `report`: deterministic compatibility
  cases, complete-group diagnostic samples, tokenizer coverage, official native
  metrics and panel metrics on the diagnostic sample. Diagnostic reports never
  expose an overall leaderboard index.
- Official reproduction kit pinned to commit
  `52a698928a9ae5bdf16b75687c903871db29c6e5` in the `benchmark` extra.
- `scripts/bench_index_cuda.sh`: CUDA headers/runtime setup and official runner.

## Results and current limitation

The published `multimodalart/decision-index-suite` download returned repository
not found. This is also reported in
https://github.com/apolinario/decision-index/issues/1 . No full-suite baseline,
official 86-request compatibility pass, or leaderboard submission is complete.

Rebuilt **23,113 requests / 32,081 fields / eight benchmarks** from the pinned
public sources using the official normalizers. This covers ContractNLI, GSM8K,
iSarcasmEval, VAST, NLI4CT, CRUXEval, CLadder and Habermas Machine. It is a partial
rebuild, not a replacement for the official corpus. Its uncompressed SHA256 is
`8dd94acf6c313c50aec3fc48188c840d994d14f764b3fff9fe7a89e2e41b9211`.

- Eight tests passed on Apple Silicon, including all 12 published choice golden
  cases and shared-prefix versus independent-readout parity.
- Compatibility: **26/26 requests succeeded**, two per available track (13 tracks).
- Full token audit of this partial corpus: **23,113/23,113 requests fit 8,192**.
  ContractNLI needs up to **8,056** tokens. Only 110/123 contracts fit 4,096;
  120/123 fit 6,144. Other rebuilt benchmarks all fit 4,096.
- `docs/bench/decision-index/coverage.json` contains coverage at several limits.
- A 500-request complete-group diagnostic run was **stopped at the user's request**
  before completion, to continue on Arch. No training was started. Its last
  progress is `artifacts/decision-index/runs/diagnostic-8k/status.json`. Do not
  confuse runner `complete` (end of supplied rows) with full-suite completion.
  `docs/bench/decision-index/diagnostic-8k.json` is an incomplete snapshot;
  inspect processed/pending counts and regenerate after a future run finishes.

## Continue on Arch / RTX 4080 Super

Connect Arch through the desktop app's Settings > Connections > SSH. Save the
same Git repository as a project there, then use this task's footer run-location
control > Arch > Hand off. The app moves the conversation and Git state. Check
that new/untracked source and report files arrived before running anything.

Ignored artifacts and `.venv` are separate from Git state. Rebuild the Linux
environment; copy `artifacts/decision-index/` explicitly over SSH if preserving
downloaded data and resumable results is desired. Do not assume these ignored
files moved with the task. Alternatively reproduce them with the commands below.
Use a NEW output directory for CUDA results: do not mix Mac and GPU timings.

```sh
uv sync --extra cuda --extra benchmark --inexact
nvidia-smi
bash scripts/bench_index_cuda.sh --help
CUDA_HOME="$PWD/.venv/cuda_home" MLX_USE_CUDA_GRAPHS=0 JET_RUN_MODEL_TESTS=1 \
  uv run --no-sync python -m unittest discover -s tests -p test_decision_index.py -v
```

The launcher prepares CUDA headers even with `--help`, and sets the MLX CUDA
cache/graph environment for benchmark runs automatically. CUDA execution has
not yet been tested on the Arch box.

Rebuild the public diagnostic corpus when not copied from the Mac:

```sh
uv run --extra benchmark python scripts/rebuild_index_diagnostic.py
uv run --extra benchmark jet-bench-index prepare \
  --suite-dir artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt \
  --out artifacts/decision-index/diagnostic --n 500 --allow-partial
```

Run compatibility before diagnostics:

```sh
bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/diagnostic/compatibility.jsonl.gz \
  --out artifacts/decision-index/runs/compatibility-cuda --option max_tokens=8192
bash scripts/bench_index_cuda.sh run --engine decision_index_engine:JetEngine \
  --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
  --out artifacts/decision-index/runs/diagnostic-cuda-8k --option max_tokens=8192
uv run --no-sync jet-bench-index report \
  --rows artifacts/decision-index/diagnostic/diagnostic.jsonl.gz \
  --results artifacts/decision-index/runs/diagnostic-cuda-8k/results.jsonl \
  --out docs/bench/decision-index/diagnostic-cuda-8k.json
```

The official runner resumes from results. Keep model revision, capacity and code
fixed within one output directory. Errors may retry; valid wrong answers do not.

When the full suite becomes available:

```sh
uv run --extra benchmark python -m decision_index suite download --dir artifacts/decision-index/suite
uv run --extra benchmark jet-bench-index prepare \
  --suite-dir artifacts/decision-index/suite --out artifacts/decision-index/full-diagnostic --n 1000
uv run --extra benchmark jet-bench-index audit \
  --rows artifacts/decision-index/suite/selected-rows.jsonl.gz \
  --out docs/bench/decision-index/full-coverage.json
```

Run the full compatibility file before launching `pipeline`. Do not disable
hash verification for a full submission. Use only the official scorer; preserve
complete source groups, exclusions, full denominators and track/macro weights.

## Next experiments

First measure retrieval and tools (not present in this partial rebuild), then
prioritize broad relevance, entailment, stance, sarcasm and tool-decision data.
Keep every evaluation row outside training. Compare new data on 0.6B first,
then the same mix on a stronger 2B–4B backbone. Calibration and latency are
secondary measurements; they do not directly improve the headline index.

Index weights: five equal areas; individual retrieval benchmarks have weight
10%, language/tools 6.67%, arts 4%, knowledge 3.33%. No training, external paid
job, publication or leaderboard submission has been started in this task.
