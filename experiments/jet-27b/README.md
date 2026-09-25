# Jet-27B experiment

Status: **specified, not trained**. No new checkpoint exists yet. The current Jet
release and all existing adapters remain intact. No compute was purchased and
nothing in this experiment was published.

## Backbone and hardware

Use Qwen/Qwen3.8-27B, pinned in experiment.json. This is a separate architecture
from the current Qwen3-0.6B model. Its hybrid attention, tokenizer, and text-only
loader require backend integration and prompt checks. Small-model LoRA tensors
cannot initialize this backbone.

The local RTX 4080 Super has 16 GB VRAM; the machine has 16 GB system RAM.
The base checkpoint is 51.75 GiB, and the published Unsloth 4-bit checkpoint is
20.81 GiB on disk. Disk size is not a measured runtime peak. Unsloth documents
24 GB VRAM for its Qwen3.8 QLoRA path. Desktop GPU use reduces available headroom
further. CPU offload with this machine's limited RAM is not a practical training
plan; experimental lower-bit methods are not validated here.

For a usable attempt, target at least a 24 GB GPU with short contexts and the
optimized documented recipe. A 48 GB GPU and 64 GB system RAM is the recommended
engineering target for more headroom, not a measured minimum. Full fine-tuning
would require much more memory than QLoRA.

## Proposed experiment

experiment.json records a proposed one-epoch rank-16 QLoRA run, LR 5e-5, batch 1,
2048-token context and gradient checkpointing. These are starting parameters,
not a validated runnable configuration. The existing train_cuda.sh does not
implement the proposed Unsloth backend or its memory optimizations.

Before training:

1. Implement the Jet label-token distribution loss on the supported text-only
   PyTorch/Unsloth loader; preserve soft targets and ordinal behavior.
2. Verify all label tokens, anonymous shuffled option keys and prompt rendering.
3. Reuse only the pinned training rows; re-audit source/content separation and
   keep selection, calibration and test splits out of optimization.
4. Benchmark the pretrained 27B baseline without changing the protocol.
5. Run a short forward/backward/checkpoint/reload smoke test and record actual
   VRAM, system RAM, tokens/second and a full-run duration estimate.
6. Train into a new directory, select on validation NLL and calibrate separately.
7. Evaluate independently with the same fixed benchmark rows as the baseline.
   Keep partial results separate from an official Decision Index score.

## Competitive potential

A stronger pretrained backbone can improve knowledge and reasoning, but it does
not guarantee Jev-level performance. Data quality, decision readout, calibration,
full prompt support, and a clean evaluation all matter. The current public
leaderboard already includes strong models using this backbone; their outcomes
are not predictions for this experiment.

## Sources

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit
- https://unsloth.ai/docs/models/qwen3.8/train

Only model metadata/configuration were fetched during preparation, not weights.
