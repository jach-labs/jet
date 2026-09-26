# Jet v6.2 release gates

Merge only the frozen lr3e-6 step-250 adapter, SHA256
53e01c1a79ec323693c1383a95240d1bf5911cbd13fc44c1e17144f4172998af,
into released Jet v6.1 revision f446b82727be57da348bb46eccf211414294ab3e.
The higher-rate trial is excluded. Verify all parent weight hashes.

Before publication, require:
- Every adapter tensor consumed and every merged tensor finite.
- Complete standalone loading with no missing/unexpected/mismatched model keys.
- Choice, score and noul wrapper smoke checks; no input truncation.
- On the previous fixed 144 merge cases plus 12 deterministic focused test cases
  and the longest complete API-Bank request: at most 3% argmax changes and maximum
  absolute probability difference below .075, using inherited temperatures.
- Re-evaluate all 914 frozen final-test examples on the standalone merged weights.
  Require the same per-family two-percentage-point regression limits and a positive
  weighted objective delta versus released v6.1. Do not select another checkpoint
  or tune merge arithmetic using these results.
- The 16,384-token runtime must handle the 11,495-token API-Bank input in full.

Benchmark scores previously displayed for v6.1 do not become v6.2 scores. The new
release evidence is the focused holdout comparison; no official Decision Index
or statistical-significance claim. Calibration is inherited and not refitted.
If any gate fails, preserve v6.1 as the current Hugging Face release and report why.
