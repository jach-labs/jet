# Jet training history

All runs below use Qwen3-0.6B unless stated otherwise. Results depend on the
specified splits and inference backend; partial benchmark samples are not official
Decision Index scores. Historical reports use “released Jet” to mean the model
published at the time of that experiment, not necessarily today's release.

## Initial experiments and V2

Public-source training introduced choice, score and yes/no decisions, followed
by varied-scale ordinal questions. V2 used two epochs and 2,910 steps, selecting
step 2,750 by validation NLL. The fused release reached about 67.7% accuracy on
the 3,683-row general test. A Qwen3-1.7B comparison was evaluated but not retained.

The complete original tables, earlier 1,500-row experiment, ordinal evaluation,
and Kev transfer evaluation are preserved in [initial results](docs/training/initial-results.md).

## V3 — broader decision training

Added relevance, entailment, stance, sarcasm, tool selection, Boolean rules,
arithmetic and code behavior. Trained on 47,313 rows for two epochs / 4,888
steps with rank-16 LoRA and learning rate `1e-4`.
New-family accuracy reached 80.6%; general-test accuracy regressed to 65.2%.

[Experiment report](docs/training/jet-v3/summary.md)

## Warm start and V4 — transfer and routing

A lower-learning-rate warm start was followed by transfer and routing training.
V4 reached 66.4% general-test accuracy and 93.3% on domain-held-out tool routing.
It improved several partial language/relevance diagnostics but regressed on code.
A CPU-fused bf16/q8 release package was prepared separately.

[Training and comparisons](docs/training/jet-next/summary.md) ·
[Export preparation](docs/training/jet-v4-release.md)

## V5 — wider intent and adversarial language tasks

Continued from V4 with 7,997 retained examples and 2,000 each from native training
partitions of CLINC150, ANLI, HellaSwag and When2Call preference data.
The completed R2 run used 15,997 rows, one epoch / 3,570 steps, rank 16,
learning rate `1e-5`, and selected step 3,500. Calibration was separate.

The first attempt was stopped after identifying retained utterances shared with
native evaluation content. R2 removes three such utterances from continuation
training, but does not erase inherited cross-dataset overlap in its initialization.
The checkpoint is not certified decontaminated.

| Held-out accuracy | V4 | V5 |
|---|---:|---:|
| General test | 66.44% | 66.66% |
| New families | 59.00% | 73.50% |
| Transfer tasks | 65.97% | 67.50% |
| Domain-held-out tool routing | 93.33% | 85.00% |

Selection-only evaluation chose original/reversed option-order averaging for the
benchmark adapter. This increases inference work and is not part of standard
serving. Gains and regressions on fixed public benchmark samples are recorded
separately; no overall leaderboard result or win has been established.

[Results](docs/training/jet-v5/summary.md) ·
[Protocol](docs/training/jet-v5/protocol.md) ·
[Completion audit](docs/training/jet-v5/completion-audit.json)

## V5 release — 2026-09-24

Superseded the same day by v6. The V5 R2 selected checkpoint was released fused
with native MLX CUDA, with a calibrated bf16 model and validated q8 ONNX export.
Its files remain at model-repository revision
`25ccbd9e09c75643b3c2214e2b2522bec39171a7`. Export and publication receipts are
recorded in [release notes](docs/training/jet-v5-release.md).

## V6 — Qwen3.5-4B — 2026-09-24

Moved the backbone to Qwen3.5-4B with a fresh rank-16 LoRA (learning rate `1e-4`)
trained with PyTorch/PEFT on the same 15,997-row `train_v5_r2` mixture, one epoch /
4,000 updates, selecting step 3,750. The adapter was merged into full bf16 weights.
Selection accuracy was 87.93% (1,400 rows) and independent local test accuracy
94.00% (600 rows). These splits differ from the V2–V5 tables above, so the numbers
are not directly comparable. The official Decision Index was not measured.

This is the current release.

[Release notes](docs/training/jet-v6-release.md) ·
[Model card](release/README.md)
