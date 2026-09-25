# Jet browser compression — 2026-09-23

Source: `michaljach/jet`, revision `8a97cfea2df622bb03f5dc9b02567e21abd2551c`.

| Export | Bytes | Accuracy | NLL | ECE |
| --- | ---: | ---: | ---: | ---: |
| Current q8 | 791,126,599 | 67.7437% | 0.87244 | 0.06162 |
| q4 candidate | 530,383,151 | 65.4629% | 0.91983 | 0.07613 |

The q4 candidate fails the predefined gates: at most 0.5 percentage points of
accuracy loss, 0.02 NLL increase, and 0.01 ECE increase. It is not approved as
the website default. Labels agree with q8 on 87.646% of the evaluation rows.
No temperatures were fitted on the test set; both models use the existing calibration.

## Evaluation method

All 3,683 rows of the versioned `data/test.jsonl`, with the browser's 2,048-token
state limit. The evaluated float32 weights were decoded exactly from each ONNX
export and executed in MLX on Metal to accelerate the full comparison. The
original input embedding, norms and rotary configuration were preserved, and
the separately quantized output head was kept untied. Native ONNX Runtime
cross-checks on 160 rows had no argmax mismatches and a maximum probability
difference below 0.000005 for either model. These cross-check rows are the
shortest prompts, so browser reference cases also cover longer inputs and all
question types. This is not a full native ONNX evaluation or a native speed benchmark.

## Compact q8 alternative

Jet only reads 267 output labels: 255 choice labels, 10 score labels, and yes/no.
The compact export removes every other output-head row, retaining the exact q8
weights/scales for all supported labels. The body and full input embedding are
unchanged. Its size is **627,326,333 bytes**, 20.7% smaller than the old export.

The output shape remains compatible with the existing browser runtime. Logits
outside the allowed label positions are placeholders and must not be used for
text generation or arbitrary vocabulary scoring. This is a typed Jet decision
export only. Structural checks compare all retained weights byte-for-byte and
native inference checks compare all 267 output label logits across all 36 golden
cases, including the long-state case. See `compact-equivalence.json` for results.

## Reproduction

Run the scripts from the repository root. Use the pinned model snapshot as
`--source`. Quantization scripts require ONNX 1.20.1, ONNX Runtime 1.30.0,
NumPy, safetensors, transformers, and PyTorch. Metal evaluation uses the existing
MLX environment. Large intermediate artifacts stay in `artifacts/quantization/`
and are excluded from Git.

- `quantize_browser.py`: symmetric int4 matrix weights, block size 32,
  rebuilt from the original bf16 weights; input embedding remains int8.
- `evaluate_browser_quantization.py`: resumable paired native CPU check.
- `dequantize_browser_for_eval.py`: exact ONNX weight decoding for evaluation.
- `evaluate_browser_metal.py`: full-set metrics and native cross-check.
- `compact_browser_head.py`: exact supported-label head compaction.
- `test_compact_browser.py`: structural and native logit equivalence.

## Browser results and deployment decision

On an Apple M2 Pro, ONNX Runtime Web 1.30.0 WebGPU, the 35 in-limit golden
reference cases had zero selected-label differences for q8 and compact q8.
The q4 candidate changed four selected labels. The compact export also retained
exactly the same maximum probability deviation from the original bf16 reference.

For the same three-question, 282-token request, after one warmup the median of
five runs was 1158.2 ms (q8), 1197.9 ms (compact q8), and 460.8 ms (q4). Compact
q8 is 3.4% slower in this small benchmark, within the 10% latency tolerance.
This is a device-specific comparison, not a general speed guarantee.

Decision: publish both artifacts, label q4 experimental, and use the compact q8
export on the website. It saves 163.8 MB without removing any supported label
weights. Original q8 and bf16 downloads remain available.
