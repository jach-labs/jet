# Released candidate training records

These records were copied from the local adapter output for the step-2,000
candidate released as Jet v6.1.0. Source paths and hashes are in provenance.json.
Absolute paths inside the records describe the original execution environment;
adapt them when reproducing the run. No weights or training examples are included.

training_config.json records the original settings and data hashes;
resume_config.json records the resumed run. adapter-config.json records the
selected adapter architecture. best-step.json records checkpoint selection.
validation-history.json preserves all 24 validation records in their original
order, including the initial model and final update.

completed.json reports 5,661 total updates. Its seconds field measures the resumed
process, not total wall time across the pause. The run resumed from step 2,250;
RNG state was not restored, so a rerun is not guaranteed to be bit-identical.
See ../resume-manifest.json and ../protocol.md for details.

The validation data informed checkpoint selection. These scores are not an
independent test or an overall Decision Index score. Benchmark comparisons and
merge validation are recorded in the neighboring evaluation and release experiments.
