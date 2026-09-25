# Training files in Git

Commit source code, run settings, dependency declarations/locks, seeds, data
provenance and hashes, split definitions, evaluation summaries and release receipts.
Keep experiment-specific records under experiments/ and explanatory history under
docs/ or TRAINING_HISTORY.md.

The released Jet v6.1 candidate has its exact run records in
[run-records](../experiments/jet-targeted-20260924/run-records/README.md), with its
[protocol](../experiments/jet-targeted-20260924/protocol.md) and
[data audit](../experiments/jet-targeted-20260924/data-audit.json) alongside them.
Historical source manifests identify original code; code-snapshot.json identifies
removed duplicate helpers and the Git blobs needed to recover them.

Weights, adapters, optimizer checkpoints, reconstructed datasets, raw prediction
outputs, logs and caches stay outside Git. Published full weights live in
[michaljach/jet](https://huggingface.co/michaljach/jet). Local adapters and
checkpoints remain in adapters/; this audit does not delete them or publish them.
Never commit credentials or tokens.

The existing data/val.jsonl, test.jsonl, score_eval.jsonl and train_v2.jsonl files
are intentional historical exceptions already tracked in Git. They were preserved
for reproducibility; new reconstructed training and benchmark rows remain ignored.

On 2026-09-25, the remaining local source edits and evaluation reports were checked
against main: the local-base adapter loader fix, its regression test, and the
release/evaluation scripts were already committed. Older local copies of helpers
and exports were superseded or covered by snapshot manifests, so they were not
reintroduced. Only missing candidate run records were added.
