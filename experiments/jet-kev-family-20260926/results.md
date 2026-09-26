# Jet v6.2 versus Kev family

Frozen transfer-v4 development clean subset: 656 questions. Higher accuracy and lower Brier are better.

| Model | Accuracy | Brier |
|---|---:|---:|
| Jev | 85.67% | 0.211 |
| Kev-27B | 84.76% | 0.236 |
| Kev-9B | 82.16% | 0.286 |
| Kev-4B | 81.71% | 0.269 |
| Jet v6.2 · 4B | 79.12% | 0.299 |
| Kev-0.8B | 64.79% | 0.481 |

| Source | Jet v6.2 | Kev-4B | Delta (pp) |
|---|---:|---:|---:|
| SciQ | 96.25% | 97.50% | -1.25 |
| QNLI | 91.25% | 91.25% | +0.00 |
| Policy: authorization | 100.00% | 100.00% | +0.00 |
| Rule: (A or B) and C | 90.62% | 100.00% | -9.38 |
| Rule: (A and B) or not C | 78.12% | 93.75% | -15.62 |
| TweetEval offensive | 72.50% | 75.00% | -2.50 |
| PAWS | 80.00% | 78.75% | +1.25 |
| Rule: if A then not B else C | 78.12% | 100.00% | -21.88 |
| MMLU, 4-way | 77.50% | 72.50% | +5.00 |
| Policy: deadline, 3-level score | 50.00% | 65.00% | -15.00 |
| Emotion, 6-way | 57.50% | 55.00% | +2.50 |

Kev/Jev are published reference results, not newly run models.
Source-out-of-domain designation applies to Kev, not Jet; Jet training includes some source families.
Development suite; not an independent final test or Decision Index score.

Reference: https://github.com/jaredpalmer/kev/blob/58d94380d4441d2d9fbb7e5e6d30d9a5b93578ae/docs/kev-family-summary.json
