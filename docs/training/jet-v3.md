# Jet v3: Decision Index data experiment

Local Arch / RTX 4080 SUPER experiment, 2026-09-23. Published Jet is unchanged.
This uses Qwen3-0.6B, not a larger backbone, and starts fresh LoRA weights from the
same frozen base. No hosted API calls, weight publication, or submission.

## Reproduction

```bash
uv sync --extra cuda --extra benchmark --inexact
uv run --no-sync python scripts/rebuild_index_diagnostic.py
uv run --no-sync python scripts/download_v3_sources.py
uv run --no-sync python -m data.decision_training
uv run --no-sync python scripts/audit_v3_data.py
uv run --no-sync python scripts/audit_v3_sources.py
bash scripts/run_v3_cuda.sh
```

The run script refuses to overwrite its completed adapter. The initial CUDA
smoke used 64 rows / 9 steps with 2,048 batch tokens. A second smoke used 128
mixed rows, including long product descriptions, at 4,096 batch tokens.
Neither smoke adapter initializes the full experiment.

## Data and isolation

`data/train_v3.jsonl` retains `train_v2` except exact held-out-state collisions
and duplicates, then adds approximately 2,000 examples per family. The original
train, validation, test and ordinal evaluation files are unchanged. Inherited
`train_v2` has no original row IDs, so its provenance is the file SHA256; new
rows carry pinned source revision, training filename, original ID (or original
row offset where the source has no ID), and a connected-component split ID.

Public training inputs:

- Product relevance: [Amazon ESCI](https://github.com/amazon-science/esci-data),
  via the pinned first training shard of `tasksource/esci`. English only,
  original example/query/product IDs and four native relevance labels.
  This is a bounded shard sample, not a representative sample of all queries.
- Document relevance: [GLUE QNLI](https://huggingface.co/datasets/nyu-mll/glue),
  training partition only, answer-containing sentence classification.
- Entailment: [SNLI](https://huggingface.co/datasets/stanfordnlp/snli), training
  partition, with missing labels rejected; shared premises stay grouped.
- Stance and irony: [TweetEval](https://github.com/cardiffnlp/tweeteval), five
  stance training partitions and the irony training partition. Irony is reported
  as irony, not relabelled as author-intended sarcasm.
- Sarcasm: [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval), English
  **training** CSV only. Author rephrases stay with their original tweets.
  All available benchmark evaluation text is excluded before splitting.
- Tool selection and relevance: [Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2),
  public synthetic training data. Use the first user request and its immediate
  function-call annotation, or an explicit single-tool refusal. No assistant
  answer or function response enters the model input. Selection adds catalog
  distractors; these are not exhaustively checked for equivalent capabilities.
  Shared native catalogs and repeated requests stay grouped. No source tools
  are executed.
- Boolean rules, arithmetic and Python code behavior: deterministic local
  generators, with labels evaluated from the generated expressions/programs.
  Only our own bounded generated code is executed. No CRUXEval/GSM8K or other
  evaluation examples generate training data. Programs with different inputs
  stay grouped; arithmetic hold-outs assess new numeric instances, not unseen
  operator families.

Source revisions and downloaded-file hashes are in
`data/train_v3.provenance.json`. Licensing follows each source: ESCI Apache-2.0,
SNLI CC-BY-SA-4.0, Glaive Apache-2.0; GLUE, TweetEval and iSarcasm retain their
upstream/component terms. This local experiment does not redistribute weights
or the collected datasets. The gated xLAM candidate was not used.

Before sampling, deduplication ignores option order. Connected groups include
source IDs, states, query/product relations, repeated tool requests/catalogs,
SNLI premises, tweets/rephrases, and generated programs/rules. Entire groups
are assigned deterministically to training, checkpoint selection, calibration,
or final evaluation. Family sampling is without replacement and round-robins
native classes. Choice order is randomized; generated numeric/output keys are
reassigned after shuffling so `option_0` cannot leak the answer.

New rows whose full state exceeds 1,024 tokens or complete prompt exceeds 2,048
are excluded. Existing train_v2 retains the prior 1,024-state-token training
policy. `selection_v3.jsonl` combines one state-grouped half of existing val
with new-family selection rows. `calibration_v3.jsonl` combines the other half
with separate new-family calibration rows. `test_v3_new.jsonl` is untouched by
checkpoint selection and temperature fitting.

The frozen Decision Index corpus remains unavailable. The rebuilt partial
corpus has 23,113 requests and SHA256
`8dd94acf6c313c50aec3fc48188c840d994d14f764b3fff9fe7a89e2e41b9211`.
It is used only for exclusion and diagnostic evaluation. A separate audit verifies all
2,000 selected product IDs, queries and labels against the original Amazon
training partition, and finds zero normalized-query overlap with all 35 pinned
ToolRet query configurations. The published
normalizer reads ESCI's official test split; our ESCI input is train. This is
not a claim that unavailable evaluation content can be exhaustively audited.

## Training and evaluation protocol

- Base: `mlx-community/Qwen3-0.6B-bf16`, pinned revision
  `42096995f6402fde107068cf530136fe64b604f8`.
- Two epochs; rank 16 on all 28 layers; LR 1e-4; LoRA scale 10; dropout .05;
  gradient checkpointing; 4,096 batch tokens; 1,024 state tokens; seed 230923.
- Preserve existing smoothing .02 and ordinal-loss weight 2.0. AdamW weight
  decay .01, warmup 100 steps followed by cosine decay to 5% of peak LR.
- Validate every 250 steps on all checkpoint-selection rows. Root adapter is
  best validation NLL; `last/` stores latest weights and optimizer state.
- Full resolved arguments, dataset hashes, validation metrics, optimizer
  checkpoints and per-type calibration are saved with the new adapter.
- Temperature fitting uses calibration only, batch size 4. Compare original
  test, ordinal evaluation, and separate new-family test against the released
  fused baseline at `8a97cfea2df622bb03f5dc9b02567e21abd2551c`.
- Run the same 500-request complete-group partial diagnostic on both models,
  preserving full prompts at an 8,192-token limit. Official native metrics
  apply only to that sample. No overall leaderboard score is reported.

All command logs are under `logs/jet-v3/`; machine-readable results are under
`docs/training/jet-v3/`. The completed scorecard is in `docs/training/jet-v3/summary.md`. All 4,888 steps
completed; the new adapter is calibrated. New-family accuracy rose from 45.9%
to 80.6%, while original test accuracy fell from 67.7% to 65.2% and ordinal
accuracy fell from 52.1% to 50.5%. Keep the released model as the default. Both
500-request partial diagnostics finished without errors; they are not leaderboard
scores. Completion checks confirmed original user edits, published model weights,
and evaluation files remain unchanged.
