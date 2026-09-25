# Jet V5 results — Qwen3-0.6B

V5 was trained locally on 15,997 examples: 7,997 retained V4 rows plus 2,000 each for CLINC150, ANLI, HellaSwag, and When2Call response preference. Native training partitions only; new selection, calibration and test groups are separate.

**These are partial diagnostics, not an official Decision Index 0.2 score or a leaderboard win.** The current board scores 40 benchmarks; the public reproduction kit still specifies the older panel. No submission was made.

R2 removes three retained short-text overlaps from continuation data. The V4 initialization still has inherited cross-dataset content overlap and is not certified decontaminated. See retention-audit.json.

Selected training step: 3500/3570; selection NLL 0.5890, accuracy 76.36%. Rank 16, LR 1e-5, one epoch, 3,072 batch tokens, 2,048 training-state tokens.

| Held-out accuracy (%) | V4 | V5 | Change (points) |
|---|---:|---:|---:|
| test | 66.44 | 66.66 | +0.22 |
| test_v5_new | 59.00 | 73.50 | +14.50 |
| test_v4_new | 65.97 | 67.50 | +1.53 |
| test_v4_tools | 93.33 | 85.00 | -8.33 |

| New held-out family | Released Jet | V4 | V5 |
|---|---:|---:|---:|
| v5:adversarial_entailment | 63.33 | 70.67 | 74.67 |
| v5:tool_response_preference | 64.00 | 58.00 | 98.00 |
| v5:commonsense_completion | 28.00 | 34.67 | 34.67 |
| v5:wide_intent | 68.67 | 72.67 | 86.67 |

## Native benchmark metrics on fixed partial samples

| Benchmark | Metric | Requests | Released Jet | V4 | V5 | V5 two-order |
|---|---|---:|---:|---:|---:|---:|
| ContractNLI | macro-F1 | 56 | 28.56 | 65.36 | 69.94 | 70.73 |
| GSM8K | accuracy | 112 | 20.54 | 37.50 | 36.61 | 50.00 |
| iSarcasmEval | see separate subtask tracks | 56 | — | — | — | — |
| VAST | macro-F1 | 56 | 36.56 | 44.31 | 40.20 | 38.66 |
| NLI4CT | macro-F1 | 55 | 25.06 | 52.03 | 53.27 | 45.00 |
| CRUXEval | accuracy | 55 | 40.00 | 32.73 | 29.09 | 36.36 |
| CLadder | accuracy | 55 | 38.18 | 47.27 | 43.64 | 50.91 |
| Habermas Machine | accuracy | 55 | 29.09 | 32.73 | 29.09 | 27.27 |
| CLINC150+OOS | macro-F1 | 200 | 14.48 | 28.18 | 34.16 | 35.34 |
| ANLI | macro-F1 | 200 | 31.47 | 36.29 | 36.17 | 33.71 |
| MMLU | accuracy | 200 | 29.50 | 30.00 | 30.50 | 33.50 |
| ARC-Easy | accuracy | 200 | 60.50 | 66.00 | 59.50 | 63.50 |
| ARC-Challenge | accuracy | 200 | 37.00 | 39.00 | 40.00 | 41.50 |
| WinoGrande | accuracy | 200 | 54.50 | 49.00 | 51.00 | 46.00 |
| HellaSwag | accuracy | 200 | 36.00 | 34.50 | 35.00 | 37.00 |

## Fixed regression checks

| Benchmark | Requests | Released Jet | V4 | V5 | V5 two-order |
|---|---:|---:|---:|---:|---:|
| code | 570 | 38.77 | 34.39 | 36.49 | 36.84 |
| esci | 500 | 19.51 | 41.35 | 38.38 | 38.61 |
| sarcasm | 1400 | 18.21 | 23.82 | 24.11 | 24.08 |

All reported values are native metrics multiplied by 100. A missing aggregate remains missing. Samples are small and results are descriptive, not statistical significance claims. MMLU and ARC are diagnostic-only and are not in the current 0.2 index panel.

Option-order policy chosen on 600 selection-only choice rows: **two_order**. The two-order engine averages original/reversed probabilities using exact option keys; it keeps complete inputs and approximately doubles forward-pass work. Selection required lower NLL without lower macro family accuracy.

The first V5 attempt was stopped because three retained utterances overlapped native evaluation content. R2 removes them from continuation data. The V4 initialization still has inherited cross-dataset content overlap; this checkpoint is not certified decontaminated. See retention-audit.json.

Single-order metrics remain separately reported. No test or benchmark score selected training steps, temperatures, or the option-order policy. Existing validation/test files and published weights were preserved.

The data protocol, source revisions and limitations are recorded in protocol.md and data/train_v5.provenance.json. Native annotations and group/source-ID separation were independently audited.
