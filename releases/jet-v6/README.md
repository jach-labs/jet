---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
language:
- en
tags:
- decision-model
- qwen3_5
- transformers
- calibrated
---

# Jet

Jet is a typed decision model based on Qwen3.5-4B. It selects from supplied
options and returns calibrated label probabilities, without generating free-form answers.
This release contains the **full merged BF16 model**. It loads directly without a separate base-model download or LoRA adapter.

**Version:** v6.0.0 · **Checkpoint:** step 3750 · **Released:** 2026-09-24.

| Question type | Input | Output |
|---|---|---|
| `choice` | 2–255 named options | Selected key and probabilities |
| `score` | 2–10 ordered levels | Expected zero-based score, selected level and probabilities |
| `noul` | Yes/no question | Probability of yes |

The model is trained for text decisions. It is not a general chat or image model.
The previous Qwen3-0.6B release is preserved under the `qwen3-0.6b-final` tag in this repository.

## Run

Tested backend: Python 3.12 on Linux, NVIDIA CUDA, PyTorch 2.11.0 + CUDA 12.8,
Transformers 5.17.0 for the included optimized runtime. Training used FP32 LoRA parameters; this
release merges them into BF16 weights. The original base revision is
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Download the versioned repository, install dependencies in a dedicated environment,
then run from that directory. Inference uses only the downloaded model files.

```sh
hf download michaljach/jet --revision v6.0.0 --local-dir jet
cd jet
python -m pip install -r requirements.txt
python jet.py <<'JSON'
{"state":"I was charged twice this month.","questions":{"topic":{"type":"choice","instructions":"What is the primary issue?","criteria":{"billing":"billing or payment problem","bug":"the product is broken"}}}}
JSON
```

Or use the included Python wrapper:

```python
from jet import Jet
model = Jet()
result = model.decide(
    "I was charged twice this month.",
    {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}},
)
print(result)
```

Use the native prompt and restricted label-token readout in these files. Generic
text-generation pipelines do not implement Jet's typed inference or calibration.
The wrapper rejects any complete question prompt over 8192 tokens rather than
truncating it. Questions are processed sequentially without shared-prefix caching.
`confidence` is normalized inverse entropy, not a separate correctness probability.
The FLA kernel is enabled; causal convolution uses the PyTorch fallback.
The previous Qwen3-0.6B MLX/ONNX exports belong to the archived release. This
release provides BF16 Transformers weights; its native typed API is shown above.

## Training

- 15,997 `train_v5_r2` examples; one epoch, 4,000 optimizer updates.
- Fresh LoRA rank/alpha 16, dropout 0.05, learning rate 1e-4, batch 1 × accumulation 4.
- Soft-target objective, label smoothing 0.02, ordinal loss weight 2; seed 240924.
- Selected step 3750 by lowest NLL on a separate 1,400-row selection split.
- Training ran on one RTX 4080 SUPER 16 GB, about 2 h 26 min; peak allocated memory 10.39 GB.
- Calibration uses a separate 1,400-row split and per-type temperature scaling.

Training and evaluation inherit Jet v5's public-source classification, ordinal and
yes/no mixtures and earlier source-exposure limitations. Source overlap has not
been comprehensively ruled out. Dataset files are not included in this release.
The backbone and pilot were selected using earlier validation results.

## Evaluation at release

| Evaluation | Result |
|---|---:|
| Selection accuracy, 1,400 rows | 87.93% |
| Selection NLL | 0.3561 |
| Independent local test accuracy, 600 rows | 94.00% |
| Local test NLL, raw / calibrated | 0.2176 / 0.2199 |
| Sampled benchmark requests | 1,900 across 15 datasets |
| Errors / unsupported requests in that diagnostic | 0 / 0 |
| Official Decision Index 0.2 overall | **Not measured** |

Calibration slightly worsened NLL on the independent local test; it was not retuned
on that test. These local results are not the overall Decision Index.

Selected diagnostic results (different case sets from the public leaderboard):

| Benchmark | Sample size | Metric | Jet score |
|---|---:|---|---:|
| GSM8K | 112 | accuracy | 76.79% |
| NLI4CT | 55 | macro-F1 | 84.01% |
| ContractNLI | 56 | macro-F1 | 74.04% |
| CRUXEval | 55 | accuracy | 38.18% |
| WinoGrande | 200 | accuracy | 64.50% |
| HellaSwag | 200 | accuracy | 91.00% |

The expanded full-benchmark evaluation is in progress at release. The current
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
requires 40 benchmarks across five areas with chance-corrected scoring. Access to
HLE and the complete matching release-v2 corpus/recipes is still required. No
subset average is presented as an overall score or rank.

Aggregate diagnostic metrics are in `evaluation.json`; `merge-provenance.json`
records the selected checkpoint and merge. `release-manifest.json` records file hashes.
`merge-validation.json` measures merged-versus-adapter behavior; the scores above
were measured on the unmerged adapter. In 36 fixed comparison cases (12 per
question type), 35 argmax answers matched; one ordinal answer changed. The largest
absolute calibrated-probability difference was 0.04373 (4.37 percentage points).
This is the expected distinction between separate LoRA inference and merging into
BF16 weights; the merged model is not claimed to be bit-identical.
Decisions can be wrong, and calibration on this mixture does not guarantee
calibration on another domain. Exact-state checks found no diagnostic/training
matches but are not comprehensive contamination certification.

## License

Apache-2.0, consistent with the base model and previous Jet release. Upstream data
sources retain their own terms. This repository does not redistribute training
or benchmark datasets.
