# Jet rule/deadline optimization

This is benchmark-informed fine-tuning of the released full Jet v6.2 model.
The baseline comparison and frozen suite provenance are in the neighboring
jet-kev-family-20260926 experiment. The published model remains unchanged.

1. Run test_generator.py, then build.py in the tokenizer/data environment.
2. Run test_metrics.py with the PyTorch/PEFT/FLA environment in environment.json.
3. launch.py verifies hashes and runs a gradient smoke test, two serial trials,
   frozen-candidate holdout evaluation, then one Kev development comparison.

See protocol.md for the exact budgets, split design, objective and regression
guards. The new rule structures and calendar-year ranges are split before fitting.
Existing banking/finance/sarcasm/retention holdouts are explicitly reused.
No benchmark examples or gold labels are used as training supervision.

Weights, optimizer states, generated rows and raw logs stay local under adapters/,
this experiment's ignored JSONL files, and logs/. status.json reports the current
phase. The scripts do not publish a candidate automatically.

The executed shared helpers are pinned by code-snapshot.json. Restore those Git
blobs when reproducing an older experiment after src/ changes. Source and input
hashes are checked before launch. Raw historical inputs are referenced by hashes
in data-audit.json and must be reconstructed from their earlier experiment records.
