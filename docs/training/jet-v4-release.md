# V4 release preparation — 2026-09-24

The training branch and locally cached `origin/main` were merged in a separate
writable checkout. The original `/home/jach/dev/jet` checkout and its model files
were preserved. Remote main could not be refreshed because network access was
unavailable. Before a GitHub push, fetch and incorporate any newer main commits.

The merge retains the Decision Index and training changes together with main's
`jet-golden` and ONNX exporter. Golden prompt rendering now uses the shared
backend-independent `inference.py`; all 36 previous reference prompts and token
sequences match. The ONNX launcher uses Python safe-path mode to avoid the local
`onnx_model.py` shadowing ONNX Runtime's module of the same name.

## Export

> The V4/V5 export and upload scripts named below (`scripts/export_release_cpu.py`,
> `run_export_cpu.py`, `upload_release.py`, `package_release.py`) were removed after the
> v6 release. They remain in git history, for example at commit `2740cca`.

Candidate: `adapters/jet-v4-transfer-20260924`, selected step 3,750. The original
adapter remains under the original checkout. New artifacts are in
`models/jet-v4` in the release checkout. No training data or previous weights
were overwritten.

CUDA initialization was unavailable during export. `scripts/export_release_cpu.py`
implements the installed MLX LoRA fusion formula on CPU, retaining bf16 weights.
It consumes all 392 adapter tensors and fuses 196 linear layers. Reference
inference uses PyTorch fp32 and the newly fused bf16 weights, with the existing
calibration. This is not evidence of CUDA/CPU numerical parity.

`scripts/run_export_cpu.py` exports fp32 ONNX with an 8 GiB resident-memory
watchdog. Actual export peak: 6,591 MiB. `src/onnx_web.py` slices logits to the last
position and performs the existing 8-bit weight-only quantization. The resulting
model is about 791 MB. Logs are under `release-logs/`; final runtime validation is
also copied into the model package as `export-validation.txt`.

The model package includes a model card, calibration, training report, run
manifest, reference vectors, fusion provenance, and per-file SHA256 manifest.
Evaluation gains and regressions are stated in its model card. Partial Decision
Index rebuild results are not an official leaderboard score.

## Upload

`scripts/upload_release.py --model models/jet-v4 --repo michaljach/jet
--receipt release-logs/huggingface-receipt.json` checks every manifest hash,
requires a passing ONNX reference check, and uploads an explicit file list in an
atomic model-repository commit with a parent-revision guard. It does not select
hardware, start a Space, or change account billing. Remote upload completion is
established only by a receipt containing the new Hugging Face commit revision.

The serving backends still pin the previous published revision. Update those
pins to the uploaded commit only after upload succeeds and before deploying the
Space. An upload alone does not update an already pinned serving endpoint.

Validation passed: all 36 q8 reference answers agree with CPU fp32. Maximum
probability difference: 0.0384; maximum label-logit difference: 0.248.
