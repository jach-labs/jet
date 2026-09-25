# Jet v6.1.0 release

This release merges the evaluated step-2,000 correction adapter into the full
merged Jet v6 parent. It does not merge against the original Qwen weights.

The executed sequence is merge.py, validate.py candidate, validate.py merged,
package.py, then publish.py. Publication validates hashes and the remote parent,
archives v6.0.0, atomically replaces main, tags v6.1.0 and verifies remote LFS hashes.
The publication receipt records the final immutable revision.

update_chart.py prepares the data/HTML for the existing jach.me GitHub Pages site;
benchmarks.js was updated to label samples and unsupported/overall missing scores.
All 26 selector views and the 360px mobile layout were checked in Chromium.

Release weights, tokenizer assets and input rows are intentionally excluded from
Git. Download the self-contained full release from michaljach/jet at v6.1.0.
Evaluation measurements are for the pre-merge adapter; the model card discloses
merge differences and the known sarcasm regression versus the previous release.
