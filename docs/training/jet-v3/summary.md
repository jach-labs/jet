# Jet v3 completed experiment

Qwen3-0.6B, Arch Linux, RTX 4080 SUPER (16 GB). Local training only; released Jet unchanged.

Completed **4,888 steps / 2 epochs**, with 47,313 training rows. Best checkpoint: step 4,888, selection NLL 0.4573, accuracy 80.6%.

New adapter: `/home/jach/dev/jet/adapters/jet-v3-decision-20260923`.
Calibration: `{"choice": 1.0, "score": 1.1225, "noul": 0.917}`.

## Independent evaluation

| Set | Rows | Released Jet accuracy | v3 accuracy | Change | Baseline → v3 NLL | Baseline → v3 ECE |
|---|---:|---:|---:|---:|---:|---:|
| test | 3683 | 67.7% | 65.2% | -2.5 pp | 0.871 → 0.937 | 0.061 → 0.089 |
| score_eval | 2400 | 52.1% | 50.5% | -1.6 pp | 1.090 → 1.109 | 0.035 → 0.050 |
| test_v3_new | 1364 | 45.9% | 80.6% | +34.8 pp | 1.401 → 0.442 | 0.283 → 0.019 |

Excluding the local Boolean, arithmetic, and code generators, the 884-row public-source subset improves from **54.0% to 71.5%**. The Glaive part is itself publicly released synthetic data.

### test: source breakdown

| Source | Rows | Released Jet | v3 | Change | NLL before → after |
|---|---:|---:|---:|---:|---:|
| ag_news | 135 | 88.1% | 89.6% | +1.5 pp | 0.304 → 0.302 |
| banking77 | 140 | 88.6% | 85.7% | -2.9 pp | 0.383 → 0.477 |
| boolq | 145 | 81.4% | 77.2% | -4.1 pp | 0.455 → 0.510 |
| civil_comments | 149 | 91.9% | 94.6% | +2.7 pp | 0.325 → 0.314 |
| dbpedia | 90 | 98.9% | 98.9% | +0.0 pp | 0.049 → 0.041 |
| emotion | 2500 | 60.3% | 56.9% | -3.4 pp | 1.062 → 1.157 |
| massive | 119 | 91.6% | 89.9% | -1.7 pp | 0.334 → 0.318 |
| mnli | 142 | 82.4% | 83.8% | +1.4 pp | 0.481 → 0.471 |
| stsb | 117 | 56.4% | 57.3% | +0.9 pp | 1.132 → 1.059 |
| yelp | 146 | 71.9% | 69.2% | -2.7 pp | 0.688 → 0.657 |

Batched latency: released fused model 6.1 ms/question; v3 unfused adapter 9.2 ms/question. This is not a fusion-controlled latency comparison.

### score_eval: source breakdown

| Source | Rows | Released Jet | v3 | Change | NLL before → after |
|---|---:|---:|---:|---:|---:|
| amazon | 600 | 47.0% | 45.3% | -1.7 pp | 1.177 → 1.223 |
| sst5 | 600 | 48.7% | 42.2% | -6.5 pp | 1.199 → 1.260 |
| stsb | 600 | 52.5% | 53.5% | +1.0 pp | 1.105 → 1.084 |
| yelp | 600 | 60.3% | 61.0% | +0.7 pp | 0.881 → 0.870 |

Batched latency: released fused model 6.5 ms/question; v3 unfused adapter 9.3 ms/question. This is not a fusion-controlled latency comparison.

Ordinal within-one accuracy: 91.6% → 90.8%; Spearman: 0.832 → 0.829; normalized MAE: 0.143 → 0.147.

### test_v3_new: source breakdown

| Source | Rows | Released Jet | v3 | Change | NLL before → after |
|---|---:|---:|---:|---:|---:|
| v3:arithmetic | 160 | 21.9% | 98.1% | +76.2 pp | 2.167 → 0.084 |
| v3:boolean_rules | 160 | 51.9% | 94.4% | +42.5 pp | 0.851 → 0.149 |
| v3:code_behavior | 160 | 19.4% | 100.0% | +80.6 pp | 1.870 → 0.022 |
| v3:document_relevance | 160 | 54.4% | 78.8% | +24.4 pp | 1.042 → 0.469 |
| v3:entailment | 160 | 66.9% | 81.2% | +14.4 pp | 0.775 → 0.501 |
| v3:product_relevance | 160 | 25.6% | 51.9% | +26.3 pp | 2.658 → 1.156 |
| v3:sarcasm_irony | 160 | 50.6% | 68.8% | +18.1 pp | 1.261 → 0.618 |
| v3:stance | 160 | 48.1% | 61.9% | +13.8 pp | 1.309 → 0.762 |
| v3:tool_relevance | 25 | 100.0% | 100.0% | +0.0 pp | 0.058 → 0.006 |
| v3:tool_selection | 59 | 100.0% | 100.0% | +0.0 pp | 0.010 → 0.013 |

Batched latency: released fused model 7.3 ms/question; v3 unfused adapter 10.2 ms/question. This is not a fusion-controlled latency comparison.

## Partial Decision Index diagnostic

**Not a leaderboard score.** Same 500 complete-group requests from the eight-benchmark public rebuild, complete native prompts up to 8,192 tokens. Frozen official suite remains unavailable.

| Benchmark | Metric | Released Jet | v3 | Requests |
|---|---|---:|---:|---:|
| ContractNLI | macro-F1 | 0.2856 | 0.2827 | 56 |
| GSM8K | accuracy | 0.2054 | 0.4643 | 112 |
| iSarcasmEval | see separate subtask tracks | not provided | not provided | 56 |
| VAST | macro-F1 | 0.3656 | 0.3232 | 56 |
| NLI4CT | macro-F1 | 0.2506 | 0.5877 | 55 |
| CRUXEval | accuracy | 0.4000 | 0.3818 | 55 |
| CLadder | accuracy | 0.3818 | 0.4364 | 55 |
| Habermas Machine | accuracy | 0.2909 | 0.2909 | 55 |

The pinned official summary provides no native iSarcasmEval subtask scores (null). Its secondary field accuracy rises from 57.3% to 66.1%, but category positive-F1 falls from 0.0317 to 0.0136; neither is substituted for a native headline score.

The synthetic code held-out gain does not transfer to CRUXEval here (40.0% → 38.2%). Likewise, the stance hold-out gain does not transfer to VAST (macro-F1 0.366 → 0.323). Keep these generator/domain limitations explicit.

Recommendation: retain released Jet as the default. Keep v3 as a data-expansion experiment; the gains on new tasks come with regressions on existing tests and mixed out-of-domain transfer. Do not tune against these final test results or promote this as a leaderboard result.

## Audit and limits

- New training examples come from pinned public training partitions and deterministic executable generators. Original IDs, hashes, grouping, and configuration are saved.
- All original data/evaluation files and published model files were preserved. 46 original training rows sharing held-out states were removed only from train_v3 (21 validation overlaps, 25 test overlaps). These are state-level overlaps, not necessarily identical questions.
- New families use separate checkpoint-selection, calibration, and final-test components. Existing validation was divided by state into selection and calibration without editing it.
- New training is balanced at 2,000 rows per family except stance (1,982). Tool tests are small after strict connected grouping; treat their changes as noisy diagnostics.
- Public Glaive annotations and distractor selection can contain semantic ambiguity. Product data uses one ESCI training shard. Synthetic tests assess these generator distributions, not general reasoning mastery.
- Training uses the existing 1,024-token state policy; overlong new rows were excluded. The Decision Index adapter preserves complete evaluation prompts.
- This single run changes the data mixture and its held-out selection/calibration mix. It does not establish a replicated causal gain or a full-suite score.

## Exact configuration

```json
{
  "base_model": "mlx-community/Qwen3-0.6B-bf16",
  "base_revision": "42096995f6402fde107068cf530136fe64b604f8",
  "train": "data/train_v3.jsonl",
  "val": "data/selection_v3.jsonl",
  "out": "adapters/jet-v3-decision-20260923",
  "epochs": 2.0,
  "lr": 0.0001,
  "warmup": 100,
  "rank": 16,
  "lora_scale": 10.0,
  "dropout": 0.05,
  "lora_layers": -1,
  "max_batch_tokens": 4096,
  "max_state_tokens": 1024,
  "smoothing": 0.02,
  "ordinal_weight": 2.0,
  "grad_checkpoint": true,
  "eval_every": 250,
  "val_limit": 10000,
  "cache_limit_gb": 1.0,
  "resume": null,
  "start_step": null,
  "seed": 230923,
  "data_sha256": {
    "data/train_v3.jsonl": "ec5666923b48289d5ab3b0887f43e52ec403c838cce7a8eee9479888ccc58be4",
    "data/selection_v3.jsonl": "f38f2fd8e503d749ea3a72a811348f663c30b9b5702f171c66f344fecc1dbee4"
  },
  "train_examples": 47313,
  "steps_per_epoch": 2444,
  "total_steps": 4888
}
```

Logs: `/home/jach/dev/jet/logs/jet-v3`. Data provenance and score JSON: `/home/jach/dev/jet/docs/training/jet-v3`.
Implementation and source descriptions: `/home/jach/dev/jet/docs/training/jet-v3.md`.
