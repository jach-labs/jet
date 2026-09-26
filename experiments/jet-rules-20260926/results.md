# Focused Jet v6.2 continuation

Validation-selected trial: lr5e-6, step 0.

| Family | Released v6.2 | Candidate |
|---|---:|---:|
| conditional | 61.25 | 61.25 |
| and_or | 56.25 | 56.25 |
| retention | 93.40 | 93.40 |
| finance | 71.21 | 71.21 |
| sarcasm | 50.00 | 50.00 |
| banking | 74.68 | 74.68 |
| or_not | 52.50 | 52.50 |
| deadline | 81.88 | 81.88 |

Keep Jet v6.2; no candidate passed the final improvement guard.

Focus holdouts are new; retention holdout is reused. No official Decision Index score is produced.

This is benchmark-informed development; targeted rule families are now included in training.
New synthetic holdouts use disjoint complete rule ASTs and date-year ranges; they share primitives and some linguistic conventions.
Banking, finance, sarcasm and broad retention holdouts are reused.
Exact-case/content screening does not establish semantic or pretraining decontamination.
Kev transfer-v4 is consulted once after checkpoint selection; its measured cases and gold labels are not used to generate supervision.

## Frozen Kev development comparison

| Task | Jet v6.2 | Candidate | Kev-4B |
|---|---:|---:|---:|
| mmlu | 77.50 | 77.50 | 72.50 |
| emotion | 57.50 | 57.50 | 55.00 |
| tweet_offensive | 72.50 | 72.50 | 75.00 |
| qnli | 91.25 | 91.25 | 91.25 |
| paws | 80.00 | 80.00 | 78.75 |
| sciq | 96.25 | 96.25 | 97.50 |
| contrastive_authorization | 100.00 | 100.00 | 100.00 |
| contrastive_deadline | 50.00 | 50.00 | 65.00 |
| composition_held_and_or | 90.62 | 90.62 | 100.00 |
| composition_held_or_not | 78.12 | 78.12 | 93.75 |
| composition_held_conditional | 78.12 | 78.12 | 100.00 |

Overall: released 79.12%, candidate 79.12%, Kev-4B 81.71%.

Benchmark-informed development: targeted families are now trained, so not an out-of-domain claim.
Kev figures are frozen published reference scores, not newly run models.
Candidate uses an unmerged correction adapter; a release requires merged-weight validation.
