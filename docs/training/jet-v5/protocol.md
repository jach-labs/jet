# Jet V5 protocol — Decision Index 0.2 coverage

Protocol recorded before V5 training and test scoring. Qwen3-0.6B remains the backbone. Initialize from the selected V4 adapter; keep all existing weights, datasets and evaluations unchanged. Use only separately sourced native training partitions for new data. Native evaluation rows are exclusion-only. Source revisions and every generated-file hash are recorded in data/train_v5.provenance.json.

New families: CLINC150 plus outside-scope intent classification with all 151 options; ANLI round-3 entailment; HellaSwag commonsense completion; When2Call chosen/rejected response preference. Related video/article, premise, tool catalog and query components stay within a split. Validation, calibration and test are separate subsets of native training partitions. Original test labels do not select checkpoints or hyperparameters. Option order is deterministic and randomized; anonymous keys do not expose preferred/rejected labels.

Retain 8,000 family-balanced V4 training rows and add up to 2,000 rows per new family. Selection and calibration each retain 800 balanced examples from their existing V4 splits and add 150 per new family. New test has 150 per family. Keep full states within 2,048 tokens and new prompts within 3,000 tokens; do not assign full-context labels to newly cropped source text.

First run a training-only 48-example smoke test. Then one epoch, all-layer LoRA rank 16, scale 10, dropout 0.05, LR 1e-5, fresh optimizer, gradient checkpointing, 3,072 batch tokens and 2,048 state tokens. Select lowest unweighted selection NLL; the initial checkpoint remains eligible. Evaluate every 250 steps and at completion. Fit temperatures on independent calibration data. Compare V4 and V5 on fixed new holdouts, original test, prior transfer/tool holdouts and the fixed partial benchmark diagnostic. Do not select based on these test results.

The live Space's 2026-09-24 snapshot describes Decision Index 0.2: 40 benchmarks in five equally weighted areas. The public reproduction kit at 8e12d6d7fead9af98155c3d23323a51144baca9d still specifies the older 19-benchmark scored panel. Its README now explains that the corpus is not redistributed and must be rebuilt due to source licensing. No partial or older-suite aggregate will be represented as the current leaderboard score.

This is one bounded improvement experiment, not a guarantee of first place. No leaderboard submission, hosted model API, or paid compute is part of this run.

## Predeclared inference ablation

After calibration, compare single-order readout with the average of original and
reversed option-order probabilities, matched by exact original key. Use a fixed,
family-balanced sample of up to 600 choice rows from selection_v5 only. Select
the two-order engine only if its NLL is lower and macro family accuracy does not
decrease. Preserve every option, full prompt and native instruction in both
passes. No benchmark identity or gold label affects inference. This doubles
forward-pass work and must be reported separately. Test labels do not choose the
inference policy. The one-order run remains a separately recorded comparison.

## R2 correction before restart

The first attempt was stopped after an independent native-source audit found
three retained short utterances overlapping native evaluation content (two
MASSIVE and one TweetEval irony example). It is not eligible for release.
R2 uses a new adapter directory and 15,997 rows, removing those examples.
Existing selection, calibration and new test sets pass this check unchanged.
The starting V4 checkpoint has inherited content overlap from legitimate native
training sources; this cannot be undone by filtering continuation data. Exact
counts and state hashes are in retention-audit.json. Do not describe R2 as fully
decontaminated, and do not claim an official leaderboard score.

Before any expanded benchmark scoring, sample size was set to 1,400 (200 per available benchmark) to improve precision. The earlier 600-row sample is preserved and is not used for the comparison.

The full available 570-case CRUXEval diagnostic is also fixed for post-training regression checking, motivated by the already-known V4 code regression. V4 reference results are reused from the prior identical rows/adapter/engine configuration; V5 single-order and selected two-order (if applicable) are scored separately.

Capacity audit: every one of the 1,400 expanded requests fits the native prompt within 4,096 tokens (maximum 1,669). The declared evaluation limit stays at 8,192, shared with previous diagnostics. No prompt truncation or option pruning is used.

V4’s earlier relevance and sarcasm gains are also regression checks: reuse the existing 500-request ESCI and all-1,400-case English-A sarcasm files without resampling. Score V5 single-order and the selected two-order policy, if applicable. Reference V4/released results use the same frozen row hashes. This is fixed before inspecting any V5 test metrics.
