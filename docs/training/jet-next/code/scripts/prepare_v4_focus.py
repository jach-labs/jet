"""Create a bounded retention mixture from V4 training rows only."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from data.decision_training import ROOT, read, write

p=ROOT/'data/train_v4_focus.jsonl'
if p.exists(): raise SystemExit('Refusing to overwrite a training mixture')
rows=read(ROOT/'data/train_v4.jsonl'); buckets=defaultdict(list); new=[]
for r in rows:
    if r['source'].startswith('v4:'):new.append(r)
    else:buckets[r['source']].append(r)
rng=random.Random(230924); selected=[]
# Round robin by original source, preserving rare families without replacement.
for rows in buckets.values():rng.shuffle(rows)
while buckets and len(selected)<18000:
    for key in list(buckets):
        selected.append(buckets[key].pop())
        if not buckets[key]:del buckets[key]
        if len(selected)==18000:break
rows=selected+new;rng.shuffle(rows);write(p,rows)
smoke=sorted([r for r in rows if r['source']=='v4:contract_entailment'],
             key=lambda r:len(r['state']['contract']),reverse=True)[:32]
smoke += [r for r in rows if r['source']=='v4:stance_transfer'][:16]
smoke += [r for r in rows if r['source']=='v4:code_transfer'][:16]
smoke_path=ROOT/'work/jet-next/v4-smoke.jsonl'
smoke_path.parent.mkdir(parents=True,exist_ok=True);write(smoke_path,smoke)
report=dict(seed=230924,retained_v3=len(selected),new_v4=len(new),total=len(rows),
    source_sha256=hashlib.sha256((ROOT/'data/train_v4.jsonl').read_bytes()).hexdigest(),
    output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    smoke_sha256=hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
    note='Without-replacement source-balanced subsample of training only; all 9000 V4 additions retained. Existing heldouts unchanged.')
(ROOT/'data/train_v4_focus.provenance.json').write_text(json.dumps(report,indent=2));print(report)
