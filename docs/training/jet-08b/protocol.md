# Jet Qwen3.5-0.8B experiment

Recorded before training/test inspection. New backbone, fresh LoRA initialization;
no 0.6B adapter is loaded. No publication or leaderboard submission.

Base: mlx-community/Qwen3.5-0.8B-MLX-bf16 at
7aef04e9adfd926ce0da9da376fe9610c8818a58 (text decoder only).
First verify architecture handling, label tokens, native inference and backward
smoke. If MLX CUDA's recurrent reference implementation is impractical, use an
isolated PyTorch CUDA backend with optimized linear attention, preserving Jet's
label-token distribution loss and original data partitions. Record any change.

Training: data/train_v5_r2.jsonl, 15,997 rows; two epochs, rank 16, LR 1e-4,
gradient checkpointing. Initial batch budget 2048 tokens; reduce for memory.
Select minimum held-out selection_v5 NLL, including the initial checkpoint as an
eligible candidate. Calibrate on calibration_v5 only. Model/optimizer/checkpoints
and logs go to new directories. The shared pretrained base is immutable.

Use all selection rows at each saved evaluation; no benchmark/test rows select
steps, learning rate, temperatures or option-order policy. Single-order is the
primary comparison. No inference-policy tuning based on benchmark results.

Compare the pretrained 0.8B baseline, trained candidate and existing Jet results
on general test, new families, transfer, routing, and ordinal decisions. If the
smoke succeeds, finish the full training run, then run the existing fixed partial
benchmark diagnostics. Preserve all existing files and their hashes.

This training set was already audited for native training partitions, grouped
splits, source IDs and content deduplication. Recheck hashes and cross-split
content before launch. The three V5 retained native-evaluation text overlaps are
absent in R2. Fresh initialization avoids inheriting Jet V4 adapter exposure,
but no claim is made about the pretrained backbone's pretraining contamination.

Private preview results remain partial diagnostics, never an official index.
