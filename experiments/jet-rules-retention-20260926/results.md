# Focused Jet v6.2 continuation

Validation-selected trial: anchor2, step 875.

| Family | Released v6.2 | Candidate |
|---|---:|---:|
| conditional | 61.25 | 63.12 |
| and_or | 56.25 | 70.00 |
| retention | 93.40 | 93.40 |
| finance | 71.21 | 73.96 |
| sarcasm | 50.00 | 51.28 |
| banking | 74.68 | 74.68 |
| or_not | 52.50 | 57.50 |
| deadline | 81.88 | 80.00 |

Collect an independent holdout, then merge and validate before any release.

All holdouts are reused development diagnostics; an independent holdout is required before release. No official Decision Index score is produced.

All splits reused from the prior rule experiment; selection and test are development diagnostics, not fresh unbiased evidence.
No validation or test examples enter supervised training or teacher caching.
A fresh independent holdout and merged-weight validation are required before release.

## Frozen Kev development comparison

| Task | Jet v6.2 | Candidate | Kev-4B |
|---|---:|---:|---:|
| mmlu | 77.50 | 77.50 | 72.50 |
| emotion | 57.50 | 57.50 | 55.00 |
| tweet_offensive | 72.50 | 70.00 | 75.00 |
| qnli | 91.25 | 91.25 | 91.25 |
| paws | 80.00 | 80.00 | 78.75 |
| sciq | 96.25 | 96.25 | 97.50 |
| contrastive_authorization | 100.00 | 100.00 | 100.00 |
| contrastive_deadline | 50.00 | 50.00 | 65.00 |
| composition_held_and_or | 90.62 | 96.88 | 100.00 |
| composition_held_or_not | 78.12 | 90.62 | 93.75 |
| composition_held_conditional | 78.12 | 87.50 | 100.00 |

Overall: released 79.12%, candidate 80.18%, Kev-4B 81.71%.

Benchmark-informed development: targeted families are now trained, so not an out-of-domain claim.
Kev figures are frozen published reference scores, not newly run models.
Candidate uses an unmerged correction adapter; a release requires merged-weight validation.
