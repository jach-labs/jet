# Jet release — 2026-09-24

Checkpoint: `jet-v5-panel02-r2-20260924`, selected step 3,500. The backbone remains
Qwen3-0.6B. Training configuration, calibration and data audits are preserved in
[the experiment directory](jet-v5/).

## Export validation

Native MLX CUDA fused 196 LoRA linear layers into bf16 weights. On 36 fixed
reference cases, fusion preserved all selected answers; the maximum probability
difference from the unfused adapter was 0.02582.

The CPU ONNX exporter used at most 6,550 MiB resident memory. The q8 export is
791 MB. Against native fused MLX CUDA reference vectors, all 36 selected answers
agree; maximum probability difference is 0.0249, maximum label-logit difference
0.254. These checks establish export compatibility on reference cases, not exact
numerical equivalence or a full benchmark evaluation of the exported model.

The model package includes calibration, reference vectors, an explicit per-file
SHA256 manifest, training reports and history. Standard serving remains
single-order; the two-order benchmark policy is separately available in source.
No leaderboard entry was submitted.

## Publication

Publication receipts and hosting status are recorded after the remote operations
complete. Hosting uses free hardware only; the account's free quota may prevent
the Space from running even after its code and weights have been uploaded.
