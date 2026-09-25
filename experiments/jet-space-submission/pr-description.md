Adds **Jet** to the **Trained** section in `news.html`, linking the source project and its full merged model. This PR targets `main` and changes only the catalog entry; it does not add an unverified leaderboard score.

### Model and inference

- Model: https://huggingface.co/michaljach/jet
- Pinned release: [`v6.0.0`, commit `e5b8f610ddb92ffaba596ae452bed32a9fef49ca`](https://huggingface.co/michaljach/jet/tree/e5b8f610ddb92ffaba596ae452bed32a9fef49ca)
- Qwen3.5-4B, trained with rank-16 LoRA on 15,997 typed-decision examples; selected checkpoint step 3750. The release contains complete merged BF16 weights, not an adapter requiring another model download.
- Native `choice` (2–255 options), ordinal `score` (2–10 levels), and `noul` (yes/no) via restricted label-token logits, with per-type temperature scaling. Complete prompts over 8,192 tokens are rejected rather than truncated.
- The release includes [`jet.py`](https://huggingface.co/michaljach/jet/blob/v6.0.0/jet.py), [`runtime.py`](https://huggingface.co/michaljach/jet/blob/v6.0.0/runtime.py), calibration, pinned requirements and usage instructions. Its Python interface is `Jet().decide(state, questions)`; no claim is made that this is a hosted `/v1/systemone` endpoint.
- Apache-2.0. The product name and canonical repository remain **Jet** / `michaljach/jet`; the earlier 0.6B release is archived under `qwen3-0.6b-final`.

### Evaluation status / request for inclusion

Please evaluate the pinned merged release for the Decision Index when possible. Its official 0.2 overall score is **not measured**. A local expanded evaluation of the frozen, unmerged checkpoint is ongoing, but the available public reproduction kit does not provide the complete matching release-v2 inputs/recipes. We are not submitting sampled results as an overall index or rank.

The model card records the completed 1,900-request diagnostic and local test results with their limits. These were measured on the unmerged checkpoint. A separate 36-case merged-versus-adapter check retained 35 argmax answers; one ordinal answer changed, with maximum probability difference 0.04373. The exact findings are public in [`merge-validation.json`](https://huggingface.co/michaljach/jet/blob/v6.0.0/merge-validation.json). The merged model should therefore be evaluated directly for leaderboard numbers.

If self-reproduction is preferred, we would appreciate the release-v2 ToolRet/BRIGHT group selection, seven extension recipes (especially PhishNChips), and current index computation inputs. HLE additionally requires dataset access on our account.

### Validation

The Space's `node check.js` passes with the new entry. Public GitHub stars (0) and Hub likes (1) were checked at submission time. No announcement post is linked because none was supplied; no engagement numbers or benchmark scores are invented.
