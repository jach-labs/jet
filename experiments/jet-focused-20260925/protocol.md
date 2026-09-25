# Focused continuation from Jet v6.1

Base: full merged BF16 michaljach/jet at immutable revision
f446b82727be57da348bb46eccf211414294ab3e. Verify local weight hashes before training.
Two serial rank/alpha-16 BF16 LoRA trials, learning rates 3e-6 and 8e-6; identical
seed 260925, mixture and update budget. No quantization. Train 1,000 updates each,
microbatch 1, accumulation 4, dropout .05, clipping 1, weight decay .01, 100-step
warmup and cosine schedule. Validate initially, every 250 updates, and at the end.
Full prompts up to 3,072 tokens; no input truncation. Run longest/p90/median prompt
gradient smoke tests first. Original base remains frozen.

4,000 examples: 800 BANKING77 train utterances with all 77 options, 800 SEntFiN
entity sentiment examples prioritizing multi-entity headlines and balancing
sentiment labels, 400 author-labeled sarcasm/literal examples at 25% positive,
and 2,000 source-balanced retention examples. Anonymous option IDs and shuffled
option order avoid fixed label-position shortcuts. No invented paraphrase labels.

Focus validation and test inputs exclude previous local training/evaluation
content. Financial headlines and sarcasm tweet/rephrase groups stay in one split.
Banking holdouts are stratified across all 77 labels. Historical benchmark inputs
are exclusion-only. Retention holdouts are reused, clearly labeled as such.
See data-audit.json for exact source revisions, hashes, realized counts and limits.

Select by 70% mean of banking accuracy, finance macro-F1 and sarcasm positive-class
F1, plus 30% retention accuracy. Every focus metric and retention accuracy must
stay within two percentage points of the initial released model. Step zero remains
the fallback. Select the winning trial before opening the new focus test. Evaluate
only the released baseline and frozen winner; do not tune on the final holdout.
A positive final objective delta and all guards are required to recommend merging.
No automatic publication; merged weights must be verified before release.

API-Bank investigation is separate: measure complete prompt lengths and smoke-test
the longest input at a 16,384-token evaluator limit. Never shorten benchmark inputs
or discard options to obtain coverage. This feasibility check produces no score.
