# Jet repair trials

Selected by validation before opening the fresh test: **lr5e-6**.

| Task / metric | Published | Previous candidate | 5e-6 | 1e-5 |
|---|---:|---:|---:|---:|
| code / accuracy | 70.15 | 78.61 | 70.15 | 70.15 |
| relevance / macro-F1 | 46.48 | 48.15 | 46.48 | 46.48 |
| sarcasm / positive-class F1 | 49.06 | 52.99 | 49.06 | 49.06 |
| preference / accuracy | 77.50 | 80.50 | 77.50 | 77.50 |
| stance / macro-F1 | 39.30 | 40.09 | 39.30 | 39.30 |

Fresh source holdouts; these are not the Decision Index benchmark scores. Published Jet is unchanged.

Exact content and group exclusions are not semantic decontamination or a pretraining contamination guarantee.
New focus holdouts are unused by previous local runs; retention validation is reused.
Generic preference examples remain a proxy for consensus; no claim of fresh Habermas testing.
Code templates are disjoint but share Python primitives.
Benchmark confusion counts informed this development run; benchmark cases never enter training.
