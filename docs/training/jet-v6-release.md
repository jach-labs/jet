# Jet v6 release — 2026-09-24

Checkpoint: step 3,750 of a fresh rank-16 LoRA on `Qwen/Qwen3.5-4B` (base revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), merged into bf16 weights. Published to
[michaljach/jet](https://huggingface.co/michaljach/jet) at commit
`e5b8f610ddb92ffaba596ae452bed32a9fef49ca`; the model card was later updated to pin
revisions by commit (`5dd7e40c`), with weights unchanged. The release tags were removed
and the repository keeps only `main`. The last Qwen3-0.6B files remain at `25ccbd9e`.

## Training

- 15,997 `train_v5_r2` examples; one epoch, 4,000 optimizer updates.
- LoRA rank/alpha 16, dropout 0.05, learning rate `1e-4`, batch 1 × accumulation 4,
  FP32 LoRA parameters.
- Soft-target objective, label smoothing 0.02, ordinal loss weight 2; seed 240924.
- Step 3,750 selected by NLL on a separate 1,400-row selection split; per-type
  temperatures ([calibration](../../releases/jet-v6/calibration.json)) fitted on another 1,400-row split.
- One RTX 4080 SUPER 16 GB, about 2 h 26 min; peak allocated memory 10.39 GB.

The training and merge code (PyTorch/PEFT) is not in this repository. The MLX
`jet-train` pipeline trains the earlier Qwen3-0.6B models only.

## Merge and validation

PEFT default merge arithmetic: bf16 backbone plus FP32 `B @ A × alpha / r`, rounded
to bf16; 248 modules, text backbone only (vision tower omitted), tied `lm_head`
([provenance](../../releases/jet-v6/merge-provenance.json)). On 36 fixed cases (12 per question
type), 35 argmax answers matched the unmerged adapter and one ordinal answer changed;
the largest calibrated-probability difference was 0.0437
([validation](../../releases/jet-v6/merge-validation.json)).

## Evaluation

Measured on the unmerged adapter:

| Evaluation | Result |
|---|---:|
| Selection accuracy / NLL (1,400 rows) | 87.93% / 0.3561 |
| Independent local test accuracy (600 rows) | 94.00% |
| Local test NLL, raw / calibrated | 0.2176 / 0.2199 |
| Sampled benchmark requests | 1,900 across 15 datasets, 0 errors or unsupported |
| Official Decision Index 0.2 overall | Not measured |

Selected diagnostics (sample sizes differ from the public leaderboard): GSM8K 76.79%
accuracy (112), NLI4CT 84.01% macro-F1 (55), ContractNLI 74.04% macro-F1 (56),
CRUXEval 38.18% accuracy (55), WinoGrande 64.50% accuracy (200), HellaSwag 91.00%
accuracy (200). Full results: [evaluation.json](../../releases/jet-v6/evaluation.json).

Training inherits v5's data mixture and its source-exposure limitations; overlap has
not been comprehensively ruled out.

## Runtime

[`releases/jet-v6/`](../../releases/jet-v6/) holds the published model card, runtime,
configuration and release records (weights and `tokenizer.json` excluded).
The runtime targets Linux + NVIDIA CUDA with PyTorch 2.11.0, Transformers 5.17.0 and
flash-linear-attention 0.5.2. It processes questions sequentially without
shared-prefix caching and rejects prompts over 8,192 tokens instead of truncating.
File hashes are in the [release manifest](../../releases/jet-v6/release-manifest.json).
