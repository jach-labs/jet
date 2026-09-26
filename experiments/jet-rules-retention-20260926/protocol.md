# Rule learning with retention anchors

Continue released full BF16 Jet v6.2, revision fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a, using fresh rank/alpha16 LoRA. Prior rule adapters are not used. Previous low-rate step500 improved sarcasm but lost 2.5 percentage points on deadline accuracy; higher-rate final weights restored deadlines but lost sarcasm. Neither passed every guard.

Two serial trials use identical data/order, learning rate 5e-6 and KL coefficients 2 and 6. Cache frozen released-model label logits on 6,000 training examples only. Forward KL uses temperature2 and T-squared scaling. Supervised loss weights: sarcasm4, deadlines3, others1. KL weight is 0.25 for the three Boolean families and1 elsewhere. This balances learning corrected labels with preserving released behavior; success is not assured.

Use unchanged audited 6,000/1,555/1,554 train/selection/diagnostic splits from jet-rules-20260926. All holdouts are reused development evidence, not fresh independent tests. Data rebuild is the parent experiment's build.py; data-audit.json pins exact files. Teacher predictions never use selection/test data. Raw examples, teacher logits and weights remain outside Git.

Each trial: 1,500 updates, microbatch1, accumulation4, 100 warmup steps, cosine schedule, AdamW weight decay0.01, gradient clipping1, seed260926, LoRA dropout0.05, maximum complete prompt3072tokens, no truncation. Evaluate selection every125steps. Objective remains70% mean rule/deadline accuracy +30% retention accuracy. All eight family metrics must remain within2percentage points of released baseline. Step-zero fallback remains; no guard is relaxed.

Freeze the validation winner before a single diagnostic evaluation and a single 656-clean-case Kev development comparison. No automatic publishing. Before promoting any candidate, require independent holdout evidence and merged-weight validation. Published Kev results are references, not rerun competitors. This is not an official Decision Index score.

launch.py verifies data, code, and full base-weight hashes, caches teachers, runs a three-example GPU forward/backward smoke check, trains both trials, then evaluates. Logs and status.json are local. code-snapshot.json pins shared source blobs used by the actual runtime; restore these in an isolated reproduction checkout when main differs.
