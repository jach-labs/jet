import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'releases/jet-v6.1'
validation=json.loads((OUT/'merge-validation.json').read_text());assert validation['passed']
comparison=json.loads((ROOT/'experiments/jet-kev-comparison-20260925/comparison.json').read_text())
(OUT/'evaluation.json').write_text(json.dumps({'evaluated_format':'Unmerged selected correction adapter on the full merged Jet v6 backbone; before final BF16 merge','candidate_step':2000,**comparison},indent=2)+'\n')
rows=['| Benchmark | Jet | Kev 8B reference |','|---|---:|---:|']
for r in comparison['comparison']:
 if r['jet_score'] is not None and r['id'] not in [2,36]:rows.append(f'| {r["dataset"]} | {100*r["jet_score"]:.2f}% | {100*r["reference_score"]:.2f}% |')
card='''---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
language:
- en
tags:
- decision-model
- qwen3_5
- transformers
---

# Jet

Jet selects typed answers from supplied options and returns probabilities without generating free-form answers.
**v6.1.0 · released 2026-09-25 · selected continuation checkpoint: step 2,000.**

This is the **full merged BF16 model**, not an adapter. No separate base-model download is needed.
It replaces v6.0.0 in the same `michaljach/jet` repository; the previous release is preserved as `v6.0.0`.
The product name remains **Jet**. Architecture: Qwen3.5-4B text-only, `Qwen3_5ForCausalLM`.

## Run

Linux + NVIDIA CUDA, Python 3.12. Tested with PyTorch 2.11.0+cu128 and Transformers 5.17.0.

```sh
hf download michaljach/jet --revision v6.1.0 --local-dir jet
cd jet
python -m pip install -r requirements.txt
python jet.py <<'JSON'
{"state":"I was charged twice this month.","questions":{"topic":{"type":"choice","instructions":"What is the primary issue?","criteria":{"billing":"billing or payment problem","bug":"the product is broken"}}}}
JSON
```

```python
from jet import Jet
model = Jet()
result = model.decide("The item arrived broken.", {
    "damaged": {"type": "noul", "instructions": "Is the item damaged?"}
})
```

| Type | Input | Output |
|---|---|---|
| `choice` | 2–255 named options | Selected key and probabilities |
| `score` | 2–10 ordered levels | Expected zero-based score, selected level, probabilities |
| `noul` | Yes/no question | Probability of yes |

Use the included native prompt and restricted label-token readout. A generic text-generation pipeline does not implement this API.
Complete prompts above 8,192 tokens are rejected; inputs and options are never silently truncated.
Questions are processed separately. The FLA kernel is enabled; convolution uses the PyTorch fallback.
`confidence` in this wrapper is normalized inverse entropy, not a separately calibrated correctness probability.
Temperatures are inherited from v6's independent calibration split; **they were not refitted or established as calibrated for this continuation**.

## Training and selection

The parent is the full merged Jet v6 release, originally pinned at
`e5b8f610ddb92ffaba596ae452bed32a9fef49ca`; exact parent shard hashes are recorded in `merge-provenance.json`.
This continuation adds a rank/alpha-16 correction LoRA to the already trained parent.
The original Qwen backbone is not used as a reset point.

- Training mixture: 22,643 examples: code, stance, sarcasm/irony, product relevance, synthetic response preferences, and broad retention tasks.
- BF16 parent / FP32 LoRA; learning rate 2e-5, dropout .05, accumulation 4, seed 24092417.
- One epoch: 5,661 updates. The selected checkpoint is **step 2,000**, not the final update.
- Selection: 1,550 examples; 70% equally weighted focus-family NLL + 30% retention NLL, with a retention guard.
- Selected validation accuracy: 88.13% versus 86.84% initially. Weighted selection NLL: 0.5455 versus 0.6288.
- Later repair trials at 5e-6 and 1e-5 failed their regression guards and are **not** included in this release.

## Benchmark evidence

The figures below measure the selected adapter **before the final BF16 merge**.
The broader run processed 67,459 requests: 66,950 answered, 509 unsupported, zero execution errors.
It covers 23 full available benchmark reconstructions and two 100-query retrieval diagnostics.
API-Bank's 508 requests exceeded the 8,192-token limit; one BRIGHT request was also unsupported.

'''+ '\n'.join(rows)+'''

Reference: archived Decision Index 0.1, frozen Space revision
`e8c3315b96e9a0b0f02cc43dd3cf62a6dc8c4316`. Kev 8B numbers are published reference scores,
not a new local Kev run. Matching metrics/counts do not prove byte-identical case sets.
Sarcasm shows English A F1; the reference additionally ran other tracks.
ToolRet (42.58) and BRIGHT (19.37) are nDCG@10 on 100 complete-query samples each, not full-suite comparisons.
**Official overall Decision Index: not measured.** No subset average is substituted for it.

This release has a known tradeoff: versus the previous published Jet, English sarcasm benchmark F1 fell
from 46.90% to 41.98%, while Habermas, VAST, code and product relevance improved in the paired local run.
The separate 951-case source-held-out diagnostic favored this candidate over v6 across its five task families;
that does not eliminate the observed benchmark regression or establish universal improvement.

## Merge verification

The correction was merged into the parent BF16 text model with FP32 B@A updates, rounded to BF16.
All 426 model tensors loaded without missing, unexpected, or mismatched keys; 248 modules received LoRA updates.
The complete export contains nine shards totaling 8,411,510,272 parameter bytes.

'''+f"On {validation['cases']} fixed verification cases, there were **{validation['argmax_flips']} changed selected answers**; the largest probability change was **{100*validation['max_probability_difference']:.2f} percentage points**. These checks do not guarantee identical benchmark scores.\n"+'''
The choice, score and noul wrapper paths were exercised. See `merge-validation.json` and `release-manifest.json`.

## Limits, sources and provenance

Benchmark aggregates informed training focus: treat this as development evaluation.
Exact content/group exclusions do not establish semantic or pretraining decontamination.
Generic synthetic response preferences are only a proxy for human consensus.
Neither model nor calibration quality is guaranteed outside the evaluated domains.

[Source and experiment records](https://github.com/michaljach/jet) ·
[Detailed benchmark report](https://github.com/michaljach/jet/blob/main/experiments/jet-kev-comparison-20260925/results.md) ·
[Charts](https://jach.me/jet/)

Apache-2.0 model/runtime; source datasets retain their own licenses. Dataset rows are not redistributed here.
'''
(OUT/'README.md').write_text(card)
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='release-manifest.json']
manifest={'name':'Jet','version':'v6.1.0','format':'complete merged BF16 Qwen3_5ForCausalLM','files':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files},'validation':{k:v for k,v in validation.items() if k not in ['records','wrapper_smoke']},'official_decision_index':None}
(OUT/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Packaged',len(files)+1,'files')
