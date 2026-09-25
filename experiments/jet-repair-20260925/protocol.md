# Jet repair trials — 2026-09-25

Two matched BF16 rank-16 LoRA trials starting from published full Jet revision
`e5b8f610ddb92ffaba596ae452bed32a9fef49ca`, not from the regressed candidate.
The only between-trial change is peak learning rate: 5e-6 versus 1e-5.
Use seed 250925, microbatch 1, accumulation 4, one shuffled pass capped at 2,000
optimizer updates, 100-step warmup, cosine decay to 5%, dropout .05, clipping 1,
weight decay .01, full prompts up to 3,072 tokens, no truncation.

The revised mixture replaces generic irony supervision with author-labeled
sarcasm and literal counterparts from the official iSarcasm training partition.
Target a 25% sarcastic training fraction before final overlap exclusions.
Retain broad decisions and code, stance, preference and relevance tasks.
Executable new code examples are separated by whole program templates.
VAST selection/test topics are absent from previous local training pools;
source article groups and repeated content are excluded across partitions.
Preference examples still measure generic response preferences, not consensus.
All source revisions, file hashes and exact realized counts are in data-audit.json.

New focus selection and fresh test examples are excluded against prior local
training, calibration, selection and test files and the reconstructed benchmark
corpus. Original/rephrased sarcasm stays grouped. Final cross-source checks remove
training rows sharing normalized content or group identities with held-out rows.
This is exact-content/group screening, not semantic or pretraining decontamination.
Retention validation is reused and is not described as fresh.

Validation occurs at step 0, every 500 updates, and the final update. Metrics:
sarcastic-class F1; stance and product-relevance macro-F1; code and preference
accuracy; broad retention accuracy. Select on 70% equally weighted mean of the
five focus metrics plus 30% retention accuracy. A checkpoint must retain sarcasm
F1 and retention accuracy within 2 percentage points of its starting model.
Step 0 remains the fallback. No calibration or decision-threshold fitting.

Freeze both checkpoints and choose the winning trial using selection metrics
before opening the fresh test for inference. Evaluate published Jet, the prior
step-2,000 candidate, and the two frozen trial checkpoints once on that test.
Do not retrain in response to test results. The final report is local; no automatic
model publication, benchmark leaderboard submission or website update.

A median/90th-percentile/longest-prompt gradient smoke test precedes the trials.
The service records progress and failures and stores checkpoints in adapters/.
