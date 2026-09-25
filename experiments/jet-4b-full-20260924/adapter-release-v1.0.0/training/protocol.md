# Full Jet Qwen3.5-4B training run

Start from Qwen/Qwen3.5-4B at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a with a fresh LoRA adapter.
The small pilot favored this backbone tentatively; its validation was used for
model selection, so no independent quality claim is made here.

Train on all 15,997 train_v5_r2 rows for one epoch: 4,000 optimizer updates,
microbatch 1, accumulation 4, rank/alpha 16, dropout .05, BF16 frozen weights,
FP32 adapter parameters, gradient checkpointing, SDPA and FLA hybrid attention.
Use peak learning rate 1e-4, 100 warmup updates, cosine decay to 5%, AdamW weight
decay .01, gradient clipping 1, seed 240924. Uniform example weights match the
pilot. Preserve Jet's soft targets, .02 smoothing and ordinal loss weight 2.
Use native prompts with thinking disabled. No quantization or prompt truncation.

Length audit: 7,825,319 training tokens, maximum 2,435; 1,400 selection rows,
maximum 2,011. Cap training at 3,072 tokens. Smoke-test median, 90th percentile
and longest training prompts before launching; then reload the original base
with a fresh seed/adapter for the actual run. The pilot already verified adapter
serialization and finite gradients. All code is snapshotted under code/.

Evaluate the full selection_v5 split initially, every 250 updates, and at the
end. Keep the checkpoint with the lowest unweighted validation NLL, including
the initial checkpoint. Save last adapter and optimizer alongside best adapter.
Calibration and test splits never enter training or checkpoint selection.
Existing source-audit and inherited contamination limitations still apply.

Outputs: adapters/jet-4b-full-20260924 (best/, last/, optimizer.pt, metrics.jsonl,
training_config.json, completed.json). Logs: logs/jet-4b-full-20260924/train.log.
status.json records the detached supervisor and training PID, phase, and failure
or completion. This run does not publish or replace the existing Jet model.
