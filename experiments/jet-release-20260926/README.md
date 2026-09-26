# Jet v6.2 release

The candidate is the frozen 3e-6 step-250 adapter from jet-focused-20260925.
The pipeline merges it into the full released v6.1 parent, verifies fixed-case
probabilities, re-evaluates all 914 final holdout examples on merged weights,
packages the self-contained model, and replaces the existing Hugging Face repo.

Executed order: run_validation.py (merge.py and both validate.py modes),
package.py, publish.py. protocol.md declares the release gates. The publication
receipt records the final remote revision and verified large-file hashes.

The standalone runtime now accepts up to 16,384 tokens. The longest reconstructed
API-Bank input passed, but no full API-Bank accuracy is claimed. The website keeps
its 25-benchmark charts attributed to v6.1 and lists v6.2 holdouts separately.
update_docs.py prepares the GitHub and website metadata; run it once per checkout.

Weights, raw cases and probability outputs are local-only, not committed to Git.
Download the full released model from michaljach/jet at tag v6.2.0.
