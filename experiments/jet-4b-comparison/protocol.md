# Jet 4B backbone pilot

Compare pinned Qwen3-4B-Instruct-2507 and Qwen3.5-4B with fresh BF16 LoRA adapters.
This is a small model-selection pilot, not an independent test or leaderboard result.

Before scoring either model, sample 20 training and 10 validation rows per source
from train_v5_r2 and selection_v5, respectively (27 sources: 540/270 rows).
Seed 240924. Require complete prompts <=2048 tokens under both tokenizers;
skip overlength candidates without truncation, record counts and hashes, and
check exact state separation. Existing source audits and inherited contamination
limitations still apply. No test or calibration rows enter this experiment.

For both models: rank 16, alpha 16, dropout .05, BF16 frozen backbone, FP32 LoRA
parameters as provided by PEFT, gradient checkpointing, SDPA, microbatch 1,
accumulation 4, AdamW with weight decay .01, gradient clipping 1, peak LR 1e-4,
10-step warmup and cosine decay to 10%, one pass over the same 540 rows (135
updates). Uniform row weights. Preserve Jet's soft-target label-token cross
entropy, .02 smoothing, and ordinal loss weight 2. Match prompt semantics using
each model's native chat template with thinking disabled. Qwen3.5 adapts hybrid
attention projections as well as attention/MLP projections; trainable parameter
counts consequently differ and are reported.

Run median/maximum-length training smoke tests and verify finite gradients and
adapter serialization before the pilots. Smoke adapters are discarded; each
pilot restarts from the original backbone and seed. Use the same software
environment and one GPU process at a time.

Measure initial and final validation NLL/accuracy, per-source results, batch-one
forward latency after warmup, training throughput, and peak PyTorch allocation.
Select initial or final by validation NLL. Compare selected models on this same
validation set, explicitly acknowledging selection bias and limited sample size.
Latency excludes model loading/tokenization, includes CPU-to-GPU input and
label readout, and synchronizes CUDA. No probability calibration is performed.
Save both initial/final predictions and adapters. Preserve existing Jet files.

Commands (from repository root):

```
.venv-qwen35/bin/python experiments/jet-4b-comparison/run.py prepare
.venv-qwen35/bin/python experiments/jet-4b-comparison/run.py smoke --model 0
.venv-qwen35/bin/python experiments/jet-4b-comparison/run.py run --model 0
.venv-qwen35/bin/python experiments/jet-4b-comparison/run.py smoke --model 1
.venv-qwen35/bin/python experiments/jet-4b-comparison/run.py run --model 1
```
