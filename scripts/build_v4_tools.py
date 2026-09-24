"""Native SGD training intent routing with whole-domain heldouts and ToolRet exclusion."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from data.decision_training import ROOT, choice, read, write, normalized, statekey, atoms, digest
from format import render_state

SEED=230924
REV='e852981ae34990f4358979625854259302feaa78'

def main():
    data=ROOT/'data'; prefix='v4_tools'
    files={s:data/p for s,p in dict(train='train_v4_tools_focus.jsonl',val='selection_v4_tools.jsonl',calibration='calibration_v4_tools.jsonl',test='test_v4_tools.jsonl').items()}
    if any(p.exists() for p in files.values()):raise SystemExit('Refusing to overwrite tool experiment')
    src=ROOT/'work/jet-next/sgd/train'
    if not src.exists():
        from decision_index.suite.build.acquire import git_fetch
        git_fetch(src.parent,'https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git',REV,['train'])
    schemas={s['service_name']:s for s in json.loads((src/'schema.json').read_text())}
    domains=sorted({s.split('_')[0] for s,v in schemas.items() if len(v['intents'])>=2})
    random.Random(SEED).shuffle(domains)
    splits={d:'val' if i<2 else 'calibration' if i<4 else 'test' if i<6 else 'train' for i,d in enumerate(domains)}
    queries_path=Path(snapshot_download('mangopy/ToolRet-Queries',repo_type='dataset',revision='b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445',allow_patterns=['*/*.parquet']))
    blocked={normalized(r['query']) for p in queries_path.rglob('*.parquet') for r in pq.read_table(p).to_pylist()}
    protected=[data/p for p in ['val.jsonl','test.jsonl','score_eval.jsonl','val_v3_new.jsonl','calibration_v3_new.jsonl','test_v3_new.jsonl','selection_v4.jsonl','calibration_v4.jsonl','test_v4_new.jsonl']]
    held_atoms={a for p in protected for r in read(p) for a in atoms(r['state'])}
    tokenizer=AutoTokenizer.from_pretrained('mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8')
    bins=defaultdict(list);excluded=Counter();seen=set();source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.json')}
    for p in sorted(src.glob('dialogues_*.json')):
        for d in json.loads(p.read_text()):
            first=d['turns'][0]
            if len(first['frames'])!=1:continue
            service=first['frames'][0]['service'];schema=schemas[service];domain=service.split('_')[0]
            if domain not in splits:continue
            # Long exact query overlap excludes the entire dialogue, without inspecting answers.
            if any(len(normalized(t['utterance']))>=32 and (normalized(t['utterance']) in blocked or digest(normalized(t['utterance'])) in held_atoms) for t in d['turns']):
                excluded['dialogue_with_evaluation_query']+=1;continue
            history=[]
            for turn,t in enumerate(d['turns'][:12]):
                history.append({'role':t['speaker'].lower(),'text':t['utterance']})
                if any(f['service']!=service for f in t['frames']):break
                if t['speaker']!='USER' or len(t['frames'])!=1:continue
                gold=t['frames'][0]['state']['active_intent']
                options={i['name']:i['description'] for i in schema['intents']}
                options['NONE']='No active intent; no service action is currently requested.'
                if gold not in options:continue
                state={'service':schema['description'],'conversation':list(history)}
                if len(tokenizer.encode(render_state(state),add_special_tokens=False))>1024:
                    excluded['over_capacity']+=1;continue
                key=statekey(state)
                if key in seen:excluded['duplicate_state']+=1;continue
                seen.add(key)
                r=choice(state,'Which service intent is active after the last user turn?',options,gold,
                    'tool_intent_routing',f'sgd:{p.name}:{d["dialogue_id"]}:{turn}',f'sgd-domain:{domain}',
                    identities=[f'sgd-dialogue:{d["dialogue_id"]}'])
                r['source']='v4:tool_intent_routing';r['provenance'].update(split=splits[domain],component=f'sgd-domain:{domain}',native_service=service,source_file=p.name)
                bins[splits[domain]].append(r)
    selected={};counts={};rng=random.Random(SEED)
    for split,rows in bins.items():
        rng.shuffle(rows);classes=defaultdict(list)
        for r in rows:classes[r['gold']].append(r)
        out=[];limit=3000 if split=='train' else 240
        while classes and len(out)<limit:
            for k in list(classes):
                out.append(classes[k].pop())
                if not classes[k]:del classes[k]
                if len(out)==limit:break
        selected[split]=out;counts[split]=len(out)
    assert all(counts.get(s,0)>=200 for s in ['val','calibration','test']),counts
    parents=dict(train=data/'train_v4_focus.jsonl',val=data/'selection_v4.jsonl',calibration=data/'calibration_v4.jsonl')
    hashes={};owners={}
    for split,rows in selected.items():
        for r in rows:
            p=r['provenance']
            for key in [p['component'],p['origin'],*p['identities'],statekey(r['state'])]:assert owners.setdefault(key,split)==split
        payload=(read(parents[split]) if split in parents else [])+rows;rng.shuffle(payload);write(files[split],payload)
        hashes[files[split].name]=hashlib.sha256(files[split].read_bytes()).hexdigest()
    report=dict(seed=SEED,repository='https://github.com/google-research-datasets/dstc8-schema-guided-dialogue',revision=REV,native_split='train',source_files=source_hashes,
        domain_splits=splits,counts=counts,exclusions=dict(excluded),toolret_query_revision='b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445',toolret_queries=len(blocked),files=hashes,
        parents={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parents.values()},code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Intent routing within one native service; not a ToolRet ranking score or BFCL multi-tool execution task.', 'Six heldout domains are entirely absent from added SGD training data; base-model pretraining coverage is unknown.', 'Exact long-query exclusion does not detect paraphrases.'])
    (data/'train_v4_tools.provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
