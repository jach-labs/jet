# Frozen 4B post-training evaluation

Evaluate step 3750 from adapters/jet-4b-full-20260924/best, already selected by
validation NLL. Do not change weights, select inference policies or tune on test
or diagnostic results. Use single-order native Jet label-token readout for all
models, complete prompts, all options, and an 8192-token limit. Unsupported
requests remain in coverage accounting. No truncation or option pruning.

Fit per-type temperatures on calibration_v5 only, using the established 121-point
log-spaced grid [.25, 8]. Fit independently for trained and untrained Qwen3.5;
report raw and calibrated NLL on separate test_v5_new. Calibration does not
change argmax accuracy. Preserve published Jet's existing calibration.

Benchmark the frozen new adapter, its original Qwen3.5-4B backbone at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, and published michaljach/jet at revision
ee7d33ecae2b8bee30891657dd57c92d34e0ce53. Use the same PyTorch single-question
readout backend to measure request latency, without shared-prefix caching.

Combine the already frozen original 500-request and expanded 1400-request
diagnostics: 1900 requests across 15 datasets. Retain complete linked groups.
Record row hashes, per-benchmark native metrics, coverage and latency with the
installed Decision Index runner/scorer pinned at
52a698928a9ae5bdf16b75687c903871db29c6e5. This is a partial diagnostic, not a full
index score. Some datasets are not part of the current scored panel. Existing
source overlap/contamination limitations apply, even where exact states differ.

The current Space reports release-v2 / Decision Index 0.2, 121057 requests,
40 scored benchmarks, corpus SHA256
b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5.
The public reproduction repository still documents the older 132422-request,
19-scored-benchmark edition. Space files contain results and methodology, not
the corpus; public repository branch/tag/PR inspection found no published 0.2
release. Save the source snapshots and do not label this diagnostic an official
0.2 score. No paid compute, publication, uploads or leaderboard submission.
