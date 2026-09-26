# Rule and deadline optimization

Start from full merged Jet v6.2, revision
fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a. Run two fresh rank/alpha-16 BF16
LoRA trials with identical data/order/seed 260926, peak learning rates 5e-6 and
1.5e-5. One epoch, 1,500 updates, microbatch 1 × accumulation 4, 100-step warmup,
cosine decay, dropout .05, weight decay .01 and gradient clipping 1. Full prompts
up to 3,072 tokens; no truncation. Validate at zero, every 250 steps and the end.

6,000 examples: 1,000 each of AND/OR compositions, negated compositions,
conditionals and three-level calendar deadline decisions; 1,000 broad replay;
302 banking, 300 entity sentiment and 398 sarcasm/literal replay examples.
Replay exclusions preserve previous heldouts and the complete frozen Kev suite.

Rules use original generators with executable truth-table labels, nested predicates,
boundary comparisons, irrelevant facts, shuffled options and counterfactual pairs.
All paired cases remain in one split. Exact ASTs are disjoint across synthetic
training/validation/test. Validation/test share primitives but use an additional
rendering style; this does not establish generalization to arbitrary policies.

Deadline supervision uses Python calendar dates and an independent integer-offset
label check. It covers inclusive deadline/grace boundaries, month/year crossings,
February and leap years. Training years are 2018–2031, validation 2033–2037, test
2038–2042. Labels are balanced approximately equally. No benchmark gold labels or
cases supply generator supervision. This is nevertheless benchmark-informed
training, and the exposed target families are now part of training.

Selection/test each have 160 fresh examples per new target family plus the
previous focused continuation's holdouts (915 validation / 914 test), explicitly
reused for banking, finance, sarcasm and retention guards.

Selection objective: 70% mean accuracy across four new target families + 30%
retention accuracy. Every target, banking accuracy, finance macro-F1, sarcasm F1
and retention accuracy must stay within two percentage points of its step-zero
baseline. Step zero is the fallback. Choose one trial by validation before opening
the new synthetic test. Evaluate released v6.2 and only that frozen candidate.

After selection and holdout evaluation, run that candidate once on the 656 clean
Kev transfer-v4 development questions. Compare with the frozen v6.2 baseline and
published Kev-4B. Report whether it matches/exceeds Kev-4B on each target and
in aggregate; do not imply statistical significance. Do not change the selected
checkpoint or train again based on those results within this experiment.

Keep the released model unchanged. A new release needs full-merge validation and
confirmation that gains and retention survive BF16 merging.
