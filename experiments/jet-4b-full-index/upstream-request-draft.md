Draft only — not sent.

Could you publish the Decision Index 0.2 reproduction recipes, or provide authorized access to its frozen evaluation inputs?

The Space at 953204c58869a05e911d5ec62b7588b1b97b03ba uses 40 scored benchmarks and balanced_skill. The public reproduction repository at 8e12d6d7fead9af98155c3d23323a51144baca9d still provides the 37-dataset / 19-scored-benchmark release-v1 kit.

We need:
1. Exact release-v2 ToolRet and BRIGHT group selection, matching the published corpus SHA256 b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5.
2. The seven extension builders, source pins and exclusions: 56 PhishNChips, 57 MMLU-Pro, 58 BBH, 59 RAGTruth, 61 HoVer, 62 When2Call, 64 New Yorker.
3. Current compute_indices.py and chance baselines, including per-track/per-query correction and ForecastBench handling.

We are evaluating a frozen Qwen3.5-4B Jet adapter locally and will not label a partial or mismatched suite as the official index. We can rebuild restricted datasets locally instead of receiving redistributed rows.
