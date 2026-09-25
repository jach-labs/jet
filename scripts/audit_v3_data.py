"""Validate v3 split isolation, executable labels, prompt limits, and immutable inputs."""
import ast
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from data.decision_training import read,statekey
from format import Question,build_prompt,render_state

root=Path(__file__).resolve().parents[1]
data=root/'data'
provenance=json.loads((data/'train_v3.provenance.json').read_text())
for name,h in provenance['protected_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
for name,h in provenance['files'].items():
    assert hashlib.sha256((data/name).read_bytes()).hexdigest()==h,name
owners={};counts={};positions=defaultdict(Counter);origins=defaultdict(set)
tokenizer=AutoTokenizer.from_pretrained('mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8')
max_state=max_prompt=checked=0
for split,file in [('train','train_v3.jsonl'),('val','val_v3_new.jsonl'),('calibration','calibration_v3_new.jsonl'),('test','test_v3_new.jsonl')]:
    rows=read(data/file);counts[split]=len(rows)
    for r in rows:
        q=Question.from_dict(r['question']);assert len(q.keys)==len(r['target'])
        assert abs(sum(r['target'])-1)<1e-6 and min(r['target'])>=0
        state=statekey(r['state'])
        assert owners.setdefault('state:'+state,split)==split
        if 'provenance' not in r:continue
        p=r['provenance'];assert p['original_split']=='train'
        for k in ['component','origin']:
            assert owners.setdefault(k+':'+p[k],split)==split,(k,p[k])
        family=r['family'];positions[family][r['target'].index(1.0)]+=1
        origins[family].add(p['origin'])
        a=len(tokenizer.encode(render_state(r['state']),add_special_tokens=False))
        b=len(tokenizer.encode(build_prompt(tokenizer,r['state'],q),add_special_tokens=False))
        max_state=max(max_state,a);max_prompt=max(max_prompt,b)
        assert a<=1024 and b<=2048
        if 'oracle' in p:
            chosen=q.criteria[q.keys[r['target'].index(1.0)]]
            if family=='boolean_rules':
                expected=bool(eval(r['state']['allow_if'],{'__builtins__':{}},r['state']['facts']))
                assert chosen==('access is allowed' if expected else 'access is denied')
            elif family=='arithmetic':
                expected=eval(r['state']['expression'],{'__builtins__':{},'abs':abs})
                assert ast.literal_eval(chosen)==expected
            else:
                ns={};exec(r['state']['code'],{'__builtins__':{},'sum':sum,'sorted':sorted,'set':set,'len':len},ns)
                assert ast.literal_eval(chosen)==ns['f'](r['state']['input'])
            checked+=1
# Selection/calibration are separate even for old val rows.
selection={statekey(r['state']) for r in read(data/'selection_v3.jsonl')}
calibration={statekey(r['state']) for r in read(data/'calibration_v3.jsonl')}
assert not selection & calibration
train={statekey(r['state']) for r in read(data/'train_v3.jsonl')}
for file in ['val.jsonl','test.jsonl','score_eval.jsonl','selection_v3.jsonl','calibration_v3.jsonl','test_v3_new.jsonl']:
    assert not train & {statekey(r['state']) for r in read(data/file)},file
report=dict(status='passed',counts=counts,max_new_state_tokens=max_state,max_new_prompt_tokens=max_prompt,
            executed_generated_labels=checked,positions={k:dict(v) for k,v in positions.items()},
            source_ids={k:len(v) for k,v in origins.items()},protected_inputs_unchanged=True)
out=root/'docs/training/jet-v3';out.mkdir(parents=True,exist_ok=True)
(out/'data-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
