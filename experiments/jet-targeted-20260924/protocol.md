# Targeted Jet continuation — 2026-09-24

User requested stopping benchmarking and further training on weak task families.
The benchmark service was stopped; this run performs training validation only.

Start from the published full merged Jet v6.0.0 at
`e5b8f610ddb92ffaba596ae452bed32a9fef49ca`, stored locally in releases/jet-v6.
Verify each base weight shard against its release manifest before loading.
Add a fresh rank/alpha-16 correction LoRA over that trained backbone. This is
continued training of the released model, not a fresh run from Qwen's original
weights. The existing published release remains unchanged.

22,643 training examples, including broader coverage from previously vetted
training pools plus 3,000 new UltraFeedback preference pairs from train_prefs.
Preference supervision is synthetic; it is a transfer task, not Habermas test
answers or new human-consensus labels. The independent 150 preference-selection
items come from prompt groups split before training selection. Keep all existing
1,400 selection_v5 rows. See data-audit.json for source revisions, hashes, counts,
protected inputs, exclusions and inherited contamination limitations.

Targets were chosen from the completed development benchmark results: CRUXEval,
Habermas, VAST, sarcasm and Amazon ESCI. These exposed benchmark results must not
be presented as fresh independent validation of this training decision.

One epoch / 5,661 updates, microbatch 1 × accumulation 4, peak LR 2e-5,
100 warmup updates and cosine decay, dropout .05, gradient clip 1, weight decay
.01, BF16 backbone and FP32 adapters, gradient checkpointing, seed 24092417.
Uniform per-example weighting after constructing the focused mixture. Preserve
native soft targets, smoothing .02 and ordinal loss weight 2. Full prompts only,
maximum 3,072 tokens; no prompt truncation.

Checkpoints every 250 updates. Selection objective: 70% equally weighted mean
NLL across the five focus families plus 30% retention NLL. A checkpoint is
eligible only if retention accuracy is at most two percentage points below its
initial value. Initial checkpoint remains the fallback. Report overall and each
family's NLL/accuracy; a selected improvement is not a guarantee of benchmark
improvement. Do not fit calibration on selection rows. No automatic publication.
