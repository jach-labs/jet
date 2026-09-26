"""Package only a full merged candidate passing the predeclared release gates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'releases/jet-v6.2'
v=json.loads((OUT/'merge-validation.json').read_text());assert v['passed']
training=json.loads((ROOT/'experiments/jet-focused-20260925/final-report.json').read_text())
evaluation={'scope':'914 fixed local holdout examples; new focus splits and reused retention split. Not official Decision Index scores.','evaluated_format':'Full merged BF16 release, using native label-token argmax','base_revision':'f446b82727be57da348bb46eccf211414294ab3e','candidate_step':250,'released_v61':v['released_holdout'],'selected_adapter':training['test']['selected_candidate'],'merged_v62':v['merged_holdout'],'test_sha256':training['test_sha256'],'limitations':training['limitations'],'official_decision_index':None}
(OUT/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
rows=['| Holdout / metric | Cases | Jet v6.1 | Jet v6.2 merged |','|---|---:|---:|---:|']
for key,label in [('banking','Banking / accuracy'),('finance','Entity sentiment / macro-F1'),('sarcasm','Sarcasm / positive-class F1'),('retention','Retention / accuracy')]:
 a=v['released_holdout']['families'][key];b=v['merged_holdout']['families'][key]
 rows.append(f'| {label} | {b["n"]} | {100*a["metric"]:.2f}% | {100*b["metric"]:.2f}% |')
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

**v6.2.0 · released 2026-09-26 · focused continuation, selected step 250.**
Jet returns typed decisions and probabilities from supplied options without generating free-form answers.
This is the **full merged BF16 Qwen3.5-4B text model**, not an adapter. No separate
base-model download is required. It replaces v6.1 in `michaljach/jet`; previous
releases remain accessible by their version tags. The product name remains Jet.

## Run

Linux + NVIDIA CUDA, Python 3.12. Verified with PyTorch 2.11.0+cu128,
Transformers 5.17.0 and flash-linear-attention 0.5.2.

```sh
hf download michaljach/jet --revision v6.2.0 --local-dir jet
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

Use the included native prompt and restricted label-token readout. A generic
text-generation pipeline does not implement this API. Questions are processed
separately. **The runtime now accepts complete prompts up to 16,384 tokens**;
inputs and options are never silently truncated. The longest reconstructed
API-Bank input (11,495 tokens) passed the standalone runtime check, but full
API-Bank accuracy has not been measured for this release. Context feasibility
does not establish long-context quality across tasks.

FLA is enabled; convolution uses the PyTorch fallback. `confidence` is normalized
inverse entropy, not a calibrated correctness probability. Temperatures are
inherited from v6.1 and were not refitted or established as calibrated for v6.2.

## Training and selection

The parent is full merged Jet v6.1 at
`f446b82727be57da348bb46eccf211414294ab3e`. Parent shard hashes and the exact
adapter hash are in `merge-provenance.json`. A fresh rank/alpha-16 correction LoRA
was trained on that parent; the model was not reset to the original Qwen weights.

- 4,000 examples: 800 banking intent, 800 entity-specific financial sentiment,
  400 sarcasm/literal, and 2,000 broad retention examples.
- BF16 backbone / FP32 adapter, dropout .05, microbatch 1, accumulation 4,
  seed 260925; two trials of 1,000 updates at peak learning rates 3e-6 and 8e-6.
- Validation-selected winner: **3e-6 at step 250**. The higher-rate trial failed
  the selection guards and is not included in this release.
- Selection used 915 examples and an objective of 70% mean focus metrics plus
  30% retention accuracy; every family must stay within two percentage points
  of its initial metric. The winner was frozen before final holdout evaluation.
- Banking uses the training partition of BANKING77, financial supervision uses
  SEntFiN, and sarcasm uses author-labeled iSarcasm training examples.
  Financial headlines and tweet/rephrase groups stay in one split.

## Full-model holdout results

The results below were re-measured on the **full merged model**, after BF16
rounding. Focus holdouts exclude the previous local input corpus listed in the
data audit; retention is reused. Small sample sizes mean these modest differences
do not establish statistical significance or broad benchmark improvement.

'''+ '\n'.join(rows)+'''

Financial sentiment is a SEntFiN transfer holdout, **not the FinEntity benchmark**.
Banking and sarcasm rows above are local source holdouts, not their public test
benchmark scores. Exact content/group exclusions do not establish semantic or
pretraining decontamination. Benchmarks informed training focus.

**Official overall Decision Index: not measured.** The 25-benchmark comparison
linked on the website belongs to the earlier v6.1 candidate, and must not be
attributed to this release. No subset average is substituted for an overall score.

## Merge verification

FP32 B@A corrections were added to the already merged parent and rounded to BF16.
All 426 tensors loaded without missing, unexpected or mismatched keys; 248 modules
received updates. Nine shards contain 8,411,510,272 parameter bytes.

'''+f"On {v['cases']} fixed verification cases, **{v['argmax_flips']} selected answers changed** between the adapter and merged model. The largest probability difference was **{100*v['max_probability_difference']:.2f} percentage points**.\n"+'''
These checks include choice, score, noul and a long API-Bank input. The full
914-case holdout passed the predeclared release guards. See evaluation.json,
merge-validation.json and release-manifest.json. Merge equivalence is approximate.

[Source and experiment records](https://github.com/michaljach/jet) ·
[Training protocol](https://github.com/michaljach/jet/blob/main/experiments/jet-focused-20260925/protocol.md) ·
[Release validation](https://github.com/michaljach/jet/blob/main/experiments/jet-release-20260926/protocol.md) ·
[Historical benchmark charts](https://jach.me/jet/)

Apache-2.0 model/runtime; source datasets retain their own licenses. Dataset rows
are not redistributed here. Quality outside the evaluated domains is not established.
'''
(OUT/'README.md').write_text(card)
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='release-manifest.json']
manifest={'name':'Jet','version':'v6.2.0','format':'complete merged BF16 Qwen3_5ForCausalLM','files':{},'validation':{k:x for k,x in v.items() if k not in ['records','wrapper_smoke','merged_holdout','released_holdout']},'official_decision_index':None}
for p in files:
 with p.open('rb') as f:manifest['files'][p.name]=hashlib.file_digest(f,'sha256').hexdigest()
(OUT/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Packaged',len(files)+1,'files',flush=True)
