# Trained Jet 4B versus Decision Index entrants

**Jet numbers below are sampled diagnostics; other columns are published full-benchmark results. These are directional comparisons, not matched-case wins or an official Jet rank.**

Jet completed all 1900 requests across 15 datasets with zero errors or unsupported requests. Fourteen datasets have directly matching scalar metric names. The frozen model is step 3750, with calibration fitted only on calibration_v5. No benchmark result was used to tune it.

| Dataset | Metric | Jet sample n | Jet 4B sample | Jev full | Decider 4B full | Kev 4B full | AutoJev-27B full |
|---|---|---:|---:|---:|---:|---:|---:|
| ContractNLI | macro-F1 | 56 | 74.0 | 71.7 | 72.3 | 62.2 | 78.1 |
| GSM8K | accuracy | 112 | 76.8 | 79.9 | 57.1 | 40.1 | 67.4 |
| VAST | macro-F1 | 56 | 43.6 | 64.6 | 53.5 | 50.9 | 70.8 |
| NLI4CT | macro-F1 | 55 | 84.0 | 84.1 | 73.7 | 72.0 | 84.8 |
| CRUXEval | accuracy | 55 | 38.2 | 73.0 | 43.7 | 46.8 | 74.9 |
| CLadder | accuracy | 55 | 65.5 | 72.6 | 63.3 | 60.2 | 74.5 |
| Habermas Machine | accuracy | 55 | 34.5 | 45.9 | 40.6 | 41.8 | 41.8 |
| CLINC150+OOS | macro-F1 | 200 | 85.3 | 89.3 | 86.6 | 79.4 | 87.8 |
| ANLI | macro-F1 | 200 | 59.6 | 74.8 | 56.4 | 46.2 | 70.7 |
| MMLU † | accuracy | 200 | 74.0 | 91.7 | 74.2 | 69.3 | 84.0 |
| ARC-Easy † | accuracy | 200 | 98.0 | 99.3 | 97.8 | 97.1 | 99.2 |
| ARC-Challenge † | accuracy | 200 | 94.0 | 97.8 | 93.9 | 89.8 | 96.8 |
| WinoGrande | accuracy | 200 | 64.5 | 92.0 | 80.7 | 68.5 | 85.2 |
| HellaSwag | accuracy | 200 | 91.0 | 94.5 | 94.0 | 75.4 | 94.0 |

Scores are displayed on a 0–100 scale. † MMLU and both ARC datasets are outside the current 0.2 scored panel. Reference denominators differ and are recorded in leaderboard-comparison.csv. The mixed iSarcasm diagnostic is omitted from the comparison because the board uses English track-A sarcasm F1; combining its subtasks would be misleading.

## What the diagnostic suggests

Jet looks competitive with similarly sized entrants on mathematics and several entailment/classification tasks. Its strongest-looking comparisons include GSM8K, ContractNLI, NLI4CT and ANLI. The clearest areas to investigate are CRUXEval code behavior, WinoGrande commonsense, VAST stance and Habermas preferences. Small samples and different case mixes prevent claims of superiority.

## Published overall Decision Index scores

| Model | Published 0.2 index |
|---|---:|
| Jet 4B | Not measured |
| Jev | 63.87 |
| Decider 4B | 52.10 |
| Kev 4B | 49.27 |
| AutoJev-27B | 63.37 |

No subset average is substituted for Jet’s missing overall score. The current index requires 40 benchmarks across five equally weighted areas; this diagnostic does not cover that panel.

## Remaining requirement for an official comparison

Obtain the matching release-v2 corpus or rebuild recipe and scoring package. The Space references 121057 requests and corpus SHA256 `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`. The public reproduction repository currently documents the older edition and does not expose the referenced lab scripts. Once the exact suite is available, verify its hash and run its complete scoring protocol.

The raw predictions, probability distributions, calibration files, code hashes, full benchmark summary and data/source audits are saved locally. No exact training-state matches were found in the diagnostic, but earlier source-contamination limitations remain; this is not a decontamination certification. No results or model weights were uploaded or submitted.

The user redirected this task away from the untrained-base comparison. That diagnostic was stopped, and the planned local published-Jet run was not started. The comparison above uses the actual Decision Index entrants’ published results.

Leaderboard snapshot generated_utc: 2026-09-24T09:01:07+00:00. Space revision: `953204c58869a05e911d5ec62b7588b1b97b03ba`.

[Decision Index leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index) · [Published data](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json) · [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json)

See leaderboard-comparison.csv for comparisons against every published entrant, including reference sample counts and metric compatibility.
