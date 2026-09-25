# Jet versus Kev 8B — broader development comparison

Reference: [archived Decision Index 0.1](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/e8c3315b96e9a0b0f02cc43dd3cf62a6dc8c4316/data/index-v0.1.json). Jet is the earlier step-2,000 candidate on the published merged backbone.

Matching metrics/counts are indicative comparisons; frozen case identity is unverified. Missing/partial results are not wins or zero scores.

| Benchmark | Jet | Kev reference | Difference (pp) | Local / reference cases | Status |
|---|---:|---:|---:|---:|---|
| iSarcasmEval | 41.98 | 38.96 | — | 1400 / 4600 | metric or denominator differs |
| VAST | 47.58 | 47.04 | +0.54 | 3006 / 3006 | complete; metric/count match |
| CRUXEval | 44.04 | 44.04 | 0.00 | 570 / 570 | complete; metric/count match |
| Habermas Machine | 44.51 | 41.71 | +2.80 | 1676 / 1676 | complete; metric/count match |
| Amazon ESCI | 52.86 | 40.23 | +12.63 | 5000 / 5000 | complete; metric/count match |
| GPQA Diamond | 43.88 | 33.16 | +10.72 | 196 / 196 | complete; metric/count match |
| MuSR | 56.91 | 54.79 | +2.12 | 752 / 752 | complete; metric/count match |
| GSM8K | 70.13 | 41.17 | +28.96 | 2638 / 2638 | complete; metric/count match |
| CLadder | 65.70 | 59.94 | +5.76 | 5000 / 5000 | complete; metric/count match |
| BFCL | 91.97 | 88.55 | +3.42 | 1694 / 1694 | complete; metric/count match |
| API-Bank | unscored | 66.73 | — | 508 / 508 | metric or denominator differs; 0 errors, 508 unsupported |
| SGD/SGD-X | 64.03 | 58.88 | +5.15 | 2500 / 2500 | complete; metric/count match |
| BANKING77 | 75.14 | 82.66 | -7.52 | 3080 / 3080 | complete; metric/count match |
| CLINC150+OOS | 91.08 | 76.79 | +14.29 | 5500 / 5500 | complete; metric/count match |
| ANLI | 62.18 | 49.31 | +12.87 | 3200 / 3200 | complete; metric/count match |
| WinoGrande | 71.03 | 68.51 | +2.52 | 1267 / 1267 | complete; metric/count match |
| HellaSwag | 92.82 | 79.50 | +13.32 | 10042 / 10042 | complete; metric/count match |
| NLI4CT | 77.17 | 70.35 | +6.82 | 5500 / 5500 | complete; metric/count match |
| ContractNLI | 76.82 | 48.52 | +28.30 | 123 / 123 | complete; metric/count match |
| BPoMP | 82.86 | 74.00 | +8.86 | 5000 / 5000 | complete; metric/count match |
| Humicroedit | 60.08 | 55.90 | +4.18 | 2628 / 2628 | complete; metric/count match |
| cfcolor | 64.36 | 57.62 | +6.74 | 5000 / 5000 | complete; metric/count match |
| FinEntity | 80.80 | 86.75 | -5.95 | 979 / 979 | complete; metric/count match |
| ToolRet | 42.58 | 42.93 | — | 100 / 7704 | sample diagnostic |
| BRIGHT | 19.37 | 14.77 | — | 100 / 1297 | sample diagnostic; 0 errors, 1 unsupported |

## Largest measured deficits

- BANKING77: -7.52 percentage points below the published reference.
- FinEntity: -5.95 percentage points below the published reference.

## Scope

- Kev figures are archived published scores, not a new local Kev inference run.
- Matching counts and metrics do not establish byte-identical cases; the official frozen corpus is unavailable.
- ToolRet and BRIGHT are 100 complete-query diagnostics, selected without outcomes; never treat their deltas as a leaderboard comparison.
- No subset average or official overall Decision Index is produced.
- Benchmarks have informed prior training choices; this is development evaluation.
