"""Update release docs while keeping v6.1 benchmark measurements explicitly historical."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];GIT=Path('/home/jach/dev/jet-release-v6-1');SITE=Path('/home/jach/dev/jach.me/src/jet')
v=json.loads((ROOT/'releases/jet-v6.2/merge-validation.json').read_text());assert v['passed']
rows=['| Local holdout | Jet v6.1 | Jet v6.2 merged |','|---|---:|---:|']
for k,label in [('banking','Banking accuracy'),('finance','Entity sentiment macro-F1'),('sarcasm','Sarcasm F1'),('retention','Retention accuracy')]:
 rows.append(f'| {label} | {100*v["released_holdout"]["families"][k]["metric"]:.2f} | {100*v["merged_holdout"]["families"][k]["metric"]:.2f} |')
section='''## Training and evaluation

Jet v6.2.0 is the full merged step-250 continuation of released Jet v6.1.
Two 1,000-update rank-16 LoRA trials used 4,000 examples, split evenly between
banking/entity sentiment/sarcasm and broad retention. Validation selected the
3e-6 trial at step 250; the higher-rate trial failed its regression guards.

The standalone merged model was evaluated on the same frozen 914-case holdout:

'''+ '\n'.join(rows)+'''

These are local holdout results, not Decision Index scores. Focus cases were
screened against previous local inputs; retention cases were reused. Financial
sentiment uses SEntFiN, not the FinEntity benchmark. The small changes do not
establish statistical significance. BF16 merging reduced the adapter's finance
F1 from 71.72 to 71.21, so only the merged scores above describe the release.

Merge checks preserved selected answers on all 157 fixed cases; maximum
probability difference was 4.76 percentage points. Calibration temperatures are
inherited, not refitted for this continuation. The complete 11,495-token API-Bank
prompt passed the new 16,384-token runtime limit without truncation; its full
benchmark score remains unmeasured.

The earlier 25-benchmark comparison measures the **v6.1 step-2,000 adapter before
its BF16 merge**, not v6.2. It includes 23 full available reconstructions and two
retrieval samples against archived Decision Index 0.1 reference scores. Matching
metrics and counts do not establish identical cases. **No official overall
Decision Index has been measured for Jet.**

[Historical v6.1 benchmark report](experiments/jet-kev-comparison-20260925/results.md) ·
[Current full-model results](releases/jet-v6.2/evaluation.json) ·
[Model card and merge verification](releases/jet-v6.2/README.md) ·
[Training protocol and records](experiments/jet-focused-20260925/README.md) ·
[Training history](TRAINING_HISTORY.md)

Release scripts and gates are recorded under
[`experiments/jet-release-20260926/`](experiments/jet-release-20260926/).
The older MLX training scripts reproduce Qwen3-0.6B generations; v6 onward uses
PyTorch/PEFT, with `src/qwen35_training.py` providing the common backend.

'''
p=GIT/'README.md';s=p.read_text();s=s[:s.index('## Training and evaluation')]+section+s[s.index('## Deployment'):]
s=s.replace('**Qwen3.5-4B** (v6.1)','**Qwen3.5-4B** (v6.2)').replace('**Jet v6.1 (Linux','**Jet v6.2 (Linux').replace('--revision v6.1.0','--revision v6.2.0').replace('releases/jet-v6.1/','releases/jet-v6.2/').replace('repository holds v6.1:','repository holds v6.2:').replace('v6.1 model card','v6.2 model card').replace('v6 CUDA runner:','v6.2 CUDA runner:').replace('up to 8,192 tokens','up to 16,384 tokens').replace('experiments/jet-targeted-20260924/run-records/README.md','experiments/jet-focused-20260925/README.md')
p.write_text(s)
p=GIT/'TRAINING_HISTORY.md';s=p.read_text();s+='''

## v6.2.0 — focused continuation, 2026-09-26

Full merged continuation from v6.1, selected at step 250 of the 3e-6 trial.
The 8e-6 trial kept its step-zero fallback and is not released. Both ran 1,000
updates on 4,000 examples: 800 banking, 800 entity sentiment, 400 sarcasm/literal,
and 2,000 retention. Protocol, data hashes, environment and trial records are in
experiments/jet-focused-20260925/.

'''+ '\n'.join(rows)+'''

These 914-case holdout measurements were repeated on the full merged weights.
Focus holdouts are new local splits; retention is reused. They are not Decision
Index scores. Financial sentiment measures SEntFiN transfer, not FinEntity.
The adapter's 71.72 financial macro-F1 became 71.21 after BF16 merging.

The 157 merge-verification cases had no argmax flips and at most 4.76 percentage
points of probability drift. The standalone runtime now accepts 16,384-token
complete prompts and passed the longest API-Bank input (11,495 tokens).
Calibration is inherited from v6.1, not refitted. Previous benchmark charts remain
explicitly attributed to v6.1; no official overall Decision Index is available.
''';p.write_text(s)
p=SITE/'index.html';s=p.read_text();s=s.replace('Jet v6.1.0 · full merged Qwen3.5-4B model. Results cover 25 local benchmarks, including two retrieval samples, alongside published model scores. Jet has no official overall index score.','Current release: Jet v6.2.0 · full merged Qwen3.5-4B model. The charts below show the earlier v6.1 evaluation across 25 benchmarks, including two retrieval samples. Jet has no official overall index score.')
s=s.replace('Jet · local evaluation','Jet v6.1 · local evaluation').replace('Scores measure the selected step-2,000 adapter before the BF16 merge. Merge validation is recorded in the model card.','Chart scores measure the v6.1 step-2,000 adapter before its BF16 merge; they are not v6.2 scores.').replace('href="https://huggingface.co/michaljach/jet">Jet v6.1.0 model card','href="https://huggingface.co/michaljach/jet/blob/v6.1.0/README.md">Jet v6.1.0 model card').replace('an 8,192-token limit','a 16,384-token limit')
html='<h3>Current release: v6.2 holdout checks</h3>\n<p class="text-small text-muted">914 local holdout examples, evaluated on the full merged model. These are separate from the benchmark charts above; financial sentiment uses SEntFiN, not FinEntity. Focus holdouts are new local splits; retention is reused. Small changes do not establish statistical significance.</p>\n<table><thead><tr><th>Holdout metric</th><th>v6.1</th><th>v6.2</th></tr></thead><tbody>'
for k,label in [('banking','Banking accuracy'),('finance','Financial sentiment F1'),('sarcasm','Sarcasm F1'),('retention','Retention accuracy')]:
 html+=f'<tr><td>{label}</td><td>{100*v["released_holdout"]["families"][k]["metric"]:.2f}</td><td>{100*v["merged_holdout"]["families"][k]["metric"]:.2f}</td></tr>'
html+='</tbody></table>\n<p class="text-small"><a href="https://huggingface.co/michaljach/jet">Current Jet model card</a> · <a href="https://github.com/michaljach/jet/blob/main/releases/jet-v6.2/evaluation.json">Full-model evaluation</a></p>\n'
s=s.replace('<h2>Question types</h2>',html+'\n<h2>Question types</h2>');p.write_text(s)
