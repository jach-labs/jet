"""Validate every SGD example against native train annotations and check isolation."""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from data.decision_training import ROOT, read, atoms, normalized, digest, statekey

src=ROOT/'work/jet-next/sgd/train'
schema={s['service_name']:s for s in json.loads((src/'schema.json').read_text())}
@lru_cache(None)
def source(name):return {d['dialogue_id']:d for d in json.loads((src/name).read_text())}
parents=read(ROOT/'data/train_v4_focus.jsonl');parent_atoms={a for r in parents for a in atoms(r['state'])}
owners={};counts=Counter();options=Counter()
for split,name in dict(train='train_v4_tools_focus.jsonl',val='selection_v4_tools.jsonl',calibration='calibration_v4_tools.jsonl',test='test_v4_tools.jsonl').items():
    for r in read(ROOT/'data'/name):
        if r['source']!='v4:tool_intent_routing':continue
        p=r['provenance'];_,filename,did,turn=p['origin'].split(':');turn=int(turn)
        d=source(filename)[did];native=d['turns'][turn]['frames'][0]
        assert native['service']==p['native_service']
        assert native['state']['active_intent']==r['gold']
        assert r['state']['conversation']==[dict(role=t['speaker'].lower(),text=t['utterance']) for t in d['turns'][:turn+1]]
        opts={i['name']:i['description'] for i in schema[p['native_service']]['intents']}
        opts['NONE']='No active intent; no service action is currently requested.'
        assert r['question']['criteria']==opts
        assert list(r['question']['criteria'])[r['target'].index(1.0)]==r['gold']
        if split!='train': assert not any(digest(normalized(t['text'])) in parent_atoms for t in r['state']['conversation'])
        for key in [p['component'],p['origin'],*p['identities'],statekey(r['state'])]:assert owners.setdefault(key,split)==split
        counts[split]+=1;options[len(opts)]+=1
report=dict(status='passed',native_annotations_checked=sum(counts.values()),counts=dict(counts),option_counts=dict(options),parent_training_overlap=0,
    checks=['Whole domains and dialogues held together','Native intent targets and complete conversation prefixes match source','Held-out conversation utterances absent from parent training content','Options are the native same-service catalog plus NONE'])
(ROOT/'docs/training/jet-next/v4-tool-audit.json').write_text(json.dumps(report,indent=2));print(report)
