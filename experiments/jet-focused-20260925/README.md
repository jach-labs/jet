# Focused Jet continuation

See protocol.md for the frozen design. This experiment trains fresh correction
adapters over the released full Jet v6.1 model, targeting banking intent,
entity-specific financial sentiment and sarcasm while replaying broad decisions.

Sources: [BANKING77](https://huggingface.co/datasets/mteb/banking77),
[SEntFiN](https://github.com/pyRis/SEntFiN), and the author-labeled training partition
of [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval). Financial sentiment is
a transfer task: no FinEntity evaluation documents supply training labels.

Run prepare_sources.py with the data environment (HF Hub, PyArrow, Transformers),
then build.py after reconstructing the historical inputs listed in data-audit.json.
The audit pins their content hashes. The training environment is in environment.json;
use a CUDA PyTorch/PEFT/FLA environment matching it. Run test_metrics.py, then
launch.py. launch.py verifies the input, source-code and base-weight hashes, runs a
gradient smoke test, trains both candidates serially, selects one using validation,
and compares it with the released model on the reserved focus test.

Artifacts: adapters/jet-focused-20260925 and logs/jet-focused-20260925 in the local
workspace. status.json records the active phase. Live status, raw rows, logits,
weights and optimizer state are excluded from Git. Retention test results reuse
historical holdouts; only the new focus holdouts are locally unexposed.

code-manifest.json hashes the executed source. code-snapshot.json identifies the
Git blobs of shared helpers used by this run; restore those when reproducing an
older experiment after shared source changes. Absolute paths in manifests identify
the original execution workspace, not a required checkout location.

API-Bank: check_context.py probes the longest complete reconstructed input at a
16,384-token limit. api-bank-context-audit.json and api-bank-context-probe.json
record coverage lengths and memory feasibility. This check does not measure
accuracy or update published benchmark charts. Reconstruct api-bank-longest.jsonl
by choosing the longest native prompt from the frozen expanded benchmark rows;
all original inputs and options must be retained.

Passing local selection is not a release. Any winning adapter still requires a
full merge, merge-equivalence checks and merged-model evaluation before publication.
