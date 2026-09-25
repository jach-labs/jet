# Full benchmark expansion

Frozen Jet Qwen3.5-4B BF16 adapter at step 3750, unchanged calibration and 8192-token complete-input policy. No untrained-base comparison. Unsupported requests remain in denominators; no truncation or pruning.

The first evaluation stages 44,922 requests from 13 current-panel benchmarks, using the complete locally rebuilt datasets and official exclusions. iSarcasm uses English track A, the current headline track. A separate source rebuild stages additional available public benchmarks; the evaluation then resumes on the expanded set without repeating existing run IDs. This work uses the public v1 recipes and must not be called a verified release-v2 overall score.

Services:
- `jet-full-index-evaluation.service`: evaluates initial.jsonl, waits for reconstruction, then evaluates expanded.jsonl and writes per-benchmark summaries.
- `jet-full-index-rebuild-v2.service`: uses the dependency-complete benchmark environment, rebuilds each independent public builder, records failures, and freezes available rows.

Status is in status.json, rebuild-status.json, and ../../artifacts/decision-index/runs/jet-4b-full-index/status.json. Logs are under ../../logs/jet-4b-full-index/. The GPU run is resumable with run.py. Both jobs are local and upload nothing.

## Requirements still preventing the current overall score

- HLE's pinned source returns HTTP 403 with configured credentials. The account needs access at https://huggingface.co/datasets/cais/hle.
- The suite repository multimodalart/decision-index-suite returns HTTP 404 with configured credentials.
- The public reproduction kit at 8e12d6d7fead9af98155c3d23323a51144baca9d lacks release-v2 selection rules for the smaller ToolRet and BRIGHT sets and builders for seven panel additions: PhishNChips, MMLU-Pro, BBH, RAGTruth, HoVer, When2Call, and New Yorker. The methodology documents several extensions but does not provide the complete frozen inputs/recipes, particularly PhishNChips.
- Obtaining the matching corpus or complete recipes is required before reporting a comparable overall score. The current headline is `balanced_skill` (chance-corrected), NOT `balanced_raw`. Jev is 51.67 versus raw 63.87 in the pinned snapshot. The home page's reference scores were corrected accordingly.

BANKING77 acquisition needed a source-path repair: the pinned HF repository has a dataset loader, not the CSVs the upstream acquirer expected. Original CSVs and categories were downloaded from the source repository referenced by that loader; revision and SHA256 are in banking77-source.json. Exact corpus equivalence remains unverified.
