# Focused Jet v6.1 continuation

Validation-selected trial: lr3e-6, step 250.

| Family | Released v6.1 | Candidate |
|---|---:|---:|
| retention | 93.40 | 93.40 |
| finance | 71.16 | 71.72 |
| banking | 74.68 | 74.68 |
| sarcasm | 46.81 | 50.00 |

Merge and validate this frozen candidate before any release.

Focus holdouts are new; retention holdout is reused. No official Decision Index score is produced.

Exact content/group exclusions do not establish semantic or pretraining decontamination.
Focus holdouts exclude previous local inputs; retention holdouts are reused.
SEntFiN measures entity-level sentiment transfer, not FinEntity benchmark performance.
Benchmarks informed target selection; final benchmark reruns are development evaluation.
No synthetic paraphrases or benchmark test labels are used as training supervision.
