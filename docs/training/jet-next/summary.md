# Jet Qwen3-0.6B follow-up results

Two more local checkpoints were trained: a lower-learning-rate warm start from released Jet, then targeted transfer training. No weights were published and no leaderboard entry was submitted.

V4 substantially improves contract entailment, product relevance, stance, and English sarcasm over released Jet on the evaluations below. Confirmed regressions remain in CRUXEval and original-test accuracy. The warm checkpoint preserves original-test accuracy more closely. Neither candidate is an across-the-board replacement.

**These results do not establish a leaderboard win.** The official frozen suite remains unavailable. Diagnostics are partial public reconstructions; the installed kit also contains an older panel definition. Native benchmark metrics below are comparable across these four models on identical rows, but are not an overall index.

| Held-out accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |
|---|---:|---:|---:|---:|
| Original test (3,683) | 67.6 | 65.2 | 67.5 | 66.4 |
| Ordinal test (2,400) | 51.7 | 50.2 | 53.0 | 52.0 |
| V3 new families (1,364) | 45.7 | 80.4 | 77.5 | 77.2 |
| Harder transfer tasks (720) | 34.2 | 50.4 | 48.3 | 66.0 |
| SGD domain-held-out routing (240) | 36.2 | 31.7 | 28.7 | 93.3 |

| Partial benchmark — native metric (%) | Released Jet | V3 | Warm start | V4 transfer |
|---|---:|---:|---:|---:|
| ContractNLI — macro-F1 | 28.6 | 28.3 | 30.8 | 65.4 |
| GSM8K — accuracy | 20.5 | 46.4 | 39.3 | 37.5 |
| iSarcasmEval — see separate subtask tracks | — | — | — | — |
| VAST — macro-F1 | 36.6 | 32.3 | 39.4 | 44.3 |
| NLI4CT — macro-F1 | 25.1 | 58.8 | 50.8 | 52.0 |
| CRUXEval — accuracy | 40.0 | 38.2 | 40.0 | 32.7 |
| CLadder — accuracy | 38.2 | 43.6 | 41.8 | 47.3 |
| Habermas Machine — accuracy | 29.1 | 29.1 | 25.5 | 32.7 |
| Amazon ESCI — macro-F1 | 19.5 | 41.4 | 41.5 | 41.3 |
| iSarcasmEval English A, all 1,400 cases — positive F1 | 18.2 | 18.9 | 17.6 | 23.8 |
| CRUXEval, all 570 available eligible cases — accuracy | 38.8 | 34.7 | 35.1 | 34.4 |

iSarcasmEval has no aggregate native score in the pinned summary report. A missing value is not zero. The separate English-A row uses the native positive-class F1 scorer on all 1,400 available cases (the official headline track), without combining other sarcasm tracks. The 500-request eight-benchmark sample and separate 500-request ESCI sample are small; differences are descriptive, not statistical significance claims.

| V4 task family — accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |
|---|---:|---:|---:|---:|
| code_transfer | 30.0 | 52.5 | 38.3 | 57.1 |
| contract_entailment | 39.6 | 45.4 | 46.7 | 80.8 |
| stance_transfer | 32.9 | 53.3 | 60.0 | 60.0 |
| tool_intent_routing | 36.2 | 31.7 | 28.7 | 93.3 |

| Original family — accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |
|---|---:|---:|---:|---:|
| ag_news | 88.1 | 89.6 | 88.9 | 88.9 |
| banking77 | 88.6 | 86.4 | 92.9 | 90.7 |
| boolq | 80.0 | 77.9 | 79.3 | 78.6 |
| civil_comments | 91.9 | 94.6 | 92.6 | 93.3 |
| dbpedia | 98.9 | 98.9 | 98.9 | 98.9 |
| emotion | 60.4 | 56.9 | 60.1 | 58.5 |
| massive | 91.6 | 89.9 | 92.4 | 93.3 |
| mnli | 81.7 | 83.8 | 80.3 | 83.1 |
| stsb | 56.4 | 57.3 | 56.4 | 54.7 |
| yelp | 71.9 | 69.2 | 69.2 | 69.9 |

| Calibration metric | Released Jet | V3 | Warm start | V4 transfer |
|---|---:|---:|---:|---:|
| test NLL | 0.8714 | 0.9363 | 0.8771 | 0.8784 |
| test ECE | 0.0622 | 0.0878 | 0.0709 | 0.0441 |
| test_v3_new NLL | 1.3997 | 0.4422 | 0.5071 | 0.5133 |
| test_v3_new ECE | 0.2851 | 0.0211 | 0.0224 | 0.0406 |
| test_v4_new NLL | 1.4212 | 1.2513 | 1.3513 | 0.7809 |
| test_v4_new ECE | 0.2345 | 0.1884 | 0.1748 | 0.0463 |
| test_v4_tools NLL | 1.7029 | 1.7665 | 1.7749 | 0.2137 |
| test_v4_tools ECE | 0.3612 | 0.3770 | 0.4560 | 0.0977 |
| Ordinal test spearman | 0.8323 | 0.8294 | 0.8367 | 0.8328 |
| Ordinal test mae | 0.1433 | 0.1473 | 0.1407 | 0.1432 |

Both candidates were selected by held-out validation NLL and calibrated on separate data before these test runs. Original three test sets use the same 4,096-token state limit for all models; new V4 tests use 2,048 for all models. Benchmark adapters preserve complete prompts up to 8,192 tokens.

## Training and provenance

- Warm start: `adapters/jet-v3-warm-20260924`; 47,313 rows, 2,444 steps, LR 2e-05, best validation step 2444, NLL 0.5131, accuracy 0.7807. Weight SHA256 `93fd8eeade6eff0d0a8aa06119f71223dc1bee36c49f1a9298d2c37816b609d4`.
- V4 transfer: `adapters/jet-v4-transfer-20260924`; 30,000 rows, 4,033 steps, LR 1e-05, best validation step 3750, NLL 0.5548, accuracy 0.7602. Weight SHA256 `4937c96fb56d7bd510f17b6aa51d5357b8bf8cc6de0e80eb976ebf47a1ea72ad`.

The published model remains unchanged. V4 retains 18,000 earlier training examples and adds 3,000 examples each for full-contract entailment, stance, executable multi-step code, and native service-intent routing. Sources are pinned; original training labels were verified; related documents/articles/domains stay together. No hosted model APIs were used.

Full experimental protocol and reproduction details follow.

# Qwen3-0.6B transfer experiment

User target: best possible local result while keeping Qwen3-0.6B. No hosted API calls, publication, or submission. This is a bounded follow-up experiment, not a claim that a leaderboard score has been beaten.

## Protocol fixed before post-training testing

1. Warm-start released Jet's local rank-16 adapter on immutable `train_v3.jsonl`, one epoch, LR 2e-5, seed 230923, 4096 batch tokens, 1024 state tokens. Fresh AdamW state and schedule. Select by `selection_v3` NLL. This ablation isolates warm-start/lower learning rate from new data.
2. Continue its best validation checkpoint on `train_v4_tools_focus.jsonl`, one epoch, LR 1e-5, seed 230924, 3072 batch tokens, 2048 state tokens. Fresh optimizer. This mixture retains 18,000 V3 rows sampled without replacement, balanced by source, and adds all 9,000 V4 transfer rows plus 3,000 native intent-routing rows. Select by `selection_v4_tools` NLL. Capped inverse-frequency weights apply only to new stance/contract classes; validation is unweighted.
3. Fit temperatures on independent calibration files. Compare both candidates with released Jet and V3 on unchanged old test sets and the 500-request partial diagnostic. Evaluate new template/document/article-held-out tests at 2048 state tokens. No test results choose a checkpoint, epoch, or temperature.

All runs use base revision `42096995f6402fde107068cf530136fe64b604f8`, all-layer LoRA rank 16, scale 10, dropout .05, gradient checkpointing, BF16 base, cache limit 1 GB. All trained checkpoint directories are new. Initial weights remain eligible if fine-tuning never improves validation; final optimizer/weights are retained separately in `last/`.

The local released adapter was compared tensor by tensor after CPU BF16 fusion against the unchanged published fused model. Tensor names match, 120 tensors match exactly, 190 differ by at most 0.00048828125 (consistent with fusion rounding). This is a close numerical reconstruction, not bit-identical fused inference. See `jet-next/baseline-adapter-check.json`.

## New data

- [VAST](https://github.com/emilyallaway/zero-shot-stance), revision `e7c4775182b184730f350995f8260579c9e066fe`, `data/VAST/vast_train.csv`. Native development/test posts are excluded. Related articles and identical documents stay together. 3,000 training examples; 240 each validation/calibration/test.
- [ContractNLI](https://github.com/stanfordnlp/contract-nli), revision `eced6528dd3c1d14d73f9a87df8f7bdbc03126f9`, archive `resources/contract-nli.zip`, member `contract-nli/train.json`. Full contracts only, fitting 2048 state tokens. Native development/test document text is excluded; one duplicate train/test document was removed. 3,000 training, 240 validation, 238 calibration, 240 test examples. These are sampled from native training documents, not the native evaluation labels.
- Local executable Python: 3,000 training, 240 each validation/calibration/test. Eighteen complete templates: twelve train, two validation, two calibration, two test. Loops, early exits, accumulators, filtering, ordering, slicing and list mutation. Counterfactual inputs supply plausible distractors. Options are shuffled and anonymous keys reassigned; every answer is checked by execution.

- [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue), revision `e852981ae34990f4358979625854259302feaa78`, native `train/` only. 3,000 training and 240 each validation/calibration/test, split by entire service domain. Native same-service intents plus NONE are the candidate set; complete dialogue prefixes determine the native active intent. All 3,720 annotations verified. Native intent routing is distinct from ToolRet ranking or BFCL execution. ToolRet public evaluation queries are exclusion-only.

Builder: `scripts/build_v4_transfer.py`; audit: `scripts/audit_v4_transfer.py`; retention mixture: `scripts/prepare_v4_focus.py`. Data hashes, source hashes, source revisions, group identities, class weights and exclusions are in `data/train_v4.provenance.json`. Existing V3, validation and test files are unchanged. V4 uses 56,313 rows before retention subsampling; initial focus mixture uses 27,000; the final tools-augmented focus mixture uses 30,000. `scripts/build_v4_tools.py` and `scripts/audit_v4_tools.py` create/audit this extension, with provenance in `data/train_v4_tools.provenance.json`.

Limitations: exact normalized matching is not semantic paraphrase detection; contracts longer than 2048 state tokens do not train; code templates share Python primitives; V3 tool tests remain small/easy and do not establish tool-ranking quality. Neither the frozen suite nor all its content is available for exclusion auditing.

## Scoring target

The official leaderboard snapshot at Space commit `eec4fe1b0a8f55eb150ae537715ff35cf211f47b` reports 19 panel benchmarks, with five equally weighted categories. The installed pinned reproduction kit's `index-panel.json` is an older proposal and differs from this deployed panel. Consequently, use native benchmark metrics from partial diagnostics, not its category totals or a reconstructed headline score.

Actual deployed panel: knowledge 24/25/30/43/44/31; language 11/40/41; retrieval 36/37; tools 1/2/6; arts 20/21/22/23/50. Snapshot archived as `jet-next/leaderboard-snapshot.json`. [Official source](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/eec4fe1b0a8f55eb150ae537715ff35cf211f47b).

The public frozen-suite dataset still returned repository-not-found at this experiment's start. Partial results cannot establish a leaderboard win.

## Reproduction

After the V3 prerequisites and sources are available:

```sh
.venv/bin/python scripts/build_v4_transfer.py
.venv/bin/python scripts/audit_v4_transfer.py
.venv/bin/python scripts/prepare_v4_focus.py
.venv/bin/python scripts/build_v4_tools.py
.venv/bin/python scripts/audit_v4_tools.py
bash train_cuda.sh --init-adapter adapters/jet \
  --base-revision 42096995f6402fde107068cf530136fe64b604f8 \
  --train data/train_v3.jsonl --val data/selection_v3.jsonl \
  --out adapters/jet-v3-warm-20260924 --epochs 1 --lr 2e-5 \
  --eval-every 250 --val-limit 10000 --seed 230923
# prepare_v4_focus also creates the 64-row training-only smoke file.
bash scripts/run_next_cuda.sh
```

Commands refuse existing dataset/checkpoint outputs. Logs are in `logs/jet-next/`; validation metrics and exact configurations are in each adapter directory. Warm-run source snapshot is preserved separately because weighted-example support was added while that already-loaded process was running.

## Additional evaluation coverage

Amazon ESCI was rebuilt in a separate directory with the pinned official normalizer and original test partition: 5,000 selected requests, uncompressed SHA256 `db1d2d3d4dde242dc04262149153bfd596044ad8986e0ae5de8443383973e386`. A deterministic complete-group 500-request sample is reserved for comparison after training. Original eight-benchmark artifacts remain unchanged. This remains a partial public reconstruction, not the frozen suite. `scripts/evaluate_next_extra.sh` compares the same rows across released Jet, V3, warm-start and V4.

The initial diagnostic contains only 19 English-A sarcasm examples. All 1,400 available English-A rows were therefore selected, solely by track ID, for a separate fixed post-training evaluation. Its metric is positive-class F1 from the official static scorer, whose headline track is `iSarcasmEval-A-En`. No aggregate score is inferred from this single track. Sample provenance is in `jet-next/isarcasm-english-sample.json`.

Evaluation context is explicit: original test/score_eval/V3 heldout data use 4096 state tokens, matching the earlier baseline inference default; all new V4 heldouts use 2048. Calibration uses 4096. Training context remains 1024 for the warm ablation and 2048 for V4. The distinction between training and inference limits is intentional and recorded.

After the fixed 55-case CRUXEval diagnostic showed a regression, evaluation was expanded to all 570 available eligible CRUXEval requests after official exclusions, across all four models. This is a targeted regression check, not another training or checkpoint-selection round. No training settings, weights, or calibration were adjusted using that result. `scripts/evaluate_next_code.sh` records these runs separately.


Final verification: 26/26 V4 compatibility requests passed; 8 adapter/GPU tests and 3 executable-data tests passed. All V4 source-annotation audits passed. Saved optimizer steps match the selected checkpoints (warm 2444, V4 3750). Original files, V3 datasets, published weights/calibration and preexisting user changes are unchanged. No training/evaluation jobs remain active.
