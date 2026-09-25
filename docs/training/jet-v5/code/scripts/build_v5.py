"""V5: native TRAIN-only coverage for the expanded Decision Index panel.

Native evaluation files are exclusion-only. All new heldouts come from source
training partitions, grouped before sampling. Never modifies earlier files.
"""
import gzip, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from data.decision_training import ROOT, Components, atoms, choice, digest, normalized, read, statekey, write
from format import Question, render_state, build_prompt
SEED=240925

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def records(spec):
    if spec['file'].endswith('.parquet'):return pq.read_table(spec['path']).to_pylist()
    return [json.loads(s) for s in open(spec['path'])]

def texts(repo,r):
    if 'clinc' in repo:return [r['text']]
    if 'hellaswag' in repo:return [r['ctx']]
    if 'anli' in repo:return [r['premise'],r['hypothesis']]
    return [r[k] for k in ['question','orig_question'] if r.get(k)]+[m['content'] for m in r.get('messages',[]) if m['role']=='user']

def content_keys(repo,r):return {'content:'+digest(normalized(t)) for t in texts(repo,r)}

def main():
    data=ROOT/'data'; work=ROOT/'work/jet-v5'
    names=['train_v5.jsonl','selection_v5.jsonl','calibration_v5.jsonl','test_v5_new.jsonl']
    assert not any((data/n).exists() for n in names),'Refusing to overwrite V5'
    specs=json.loads((work/'sources-complete.json').read_text())
    protected=[p for p in data.glob('*.jsonl') if any(v in p.name for v in ['val','test','selection','calibration'])]
    protected+=[ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl',ROOT/'artifacts/decision-index/esci-rebuild/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl']
    blocked=set(); old_atoms=set(); counts=Counter(); native_groups=set()
    for p in protected:
        assert p.exists(),p
        for r in read(p):
            blocked.update('content:'+a for a in atoms(r.get('state',r.get('payload',{}))))
            if isinstance(r.get('state'),str):blocked.add('content:'+digest(normalized(r['state'])))
    parent=read(data/'train_v4_tools_focus.jsonl')
    # Include all earlier training, even examples not retained in this short stage.
    for p in [data/'train_v2.jsonl',data/'train_v3.jsonl',data/'train_v4.jsonl',data/'train_v4_tools_focus.jsonl']:
        for r in read(p):
            old_atoms.update('content:'+a for a in atoms(r['state']))
            if isinstance(r['state'],str):old_atoms.add('content:'+digest(normalized(r['state'])))
    for s in specs:
        if s['role']=='train':continue
        for r in records(s):
            blocked.update(content_keys(s['repo'],r))
            if 'hellaswag' in s['repo']:native_groups.add('hellaswag:'+r['source_id'])
    candidates=[]; components=Components(); origin_seen=set()
    for s in specs:
        if s['role']!='train':continue
        repo=s['repo']; rows=records(s)
        if 'clinc' in repo:
            labels=json.loads(pq.read_schema(s['path']).metadata[b'huggingface'])['info']['features']['intent']['names']
            opts={k:('outside all supported intents' if k=='oos' else k.replace('_',' ')) for k in labels}
        if 'anli' in repo:
            assert json.loads(pq.read_schema(s['path']).metadata[b'huggingface'])['info']['features']['label']['names']==['entailment','neutral','contradiction']
        for idx,r in enumerate(rows):
            keys=content_keys(repo,r)
            if keys&blocked:counts['evaluation_content']+=1;continue
            origin=f'{repo}@{s["revision"]}/{s["file"]}:{r.get("uid",r.get("ind",idx))}'
            if origin in origin_seen:counts['duplicate_source_id']+=1;continue
            origin_seen.add(origin)
            if 'clinc' in repo:
                family='wide_intent'; group='clinc:'+digest(normalized(r['text']))
                row=choice(r['text'],'Which intent best describes the user request? Choose outside scope if none apply.',opts,labels[r['intent']],family,origin,group)
            elif 'hellaswag' in repo:
                family='commonsense_completion';group='hellaswag:'+r['source_id']
                if group in native_groups:counts['native_eval_video_or_article']+=1;continue
                options={str(i):v for i,v in enumerate(r['endings'])}
                row=choice(r['ctx'],'Which continuation is most plausible?',options,str(int(r['label'])),family,origin,group)
            elif 'anli' in repo:
                family='adversarial_entailment';group='anli:'+digest(normalized(r['premise']))
                options=dict(entailment='the premise entails the hypothesis',neutral='the premise does not determine the hypothesis',contradiction='the premise contradicts the hypothesis')
                row=choice(dict(premise=r['premise'],hypothesis=r['hypothesis']),'How does the hypothesis relate to the premise?',options,list(options)[r['label']],family,origin,group)
            else:
                family='tool_response_preference'; tools=[json.loads(t) if isinstance(t,str) else t for t in r['tools']]
                group='when2call-tools:'+digest(sorted(t['name'] for t in tools))
                options=dict(preferred=r['chosen_response']['content'],rejected=r['rejected_response']['content'])
                if options['preferred']==options['rejected']:counts['identical_choices']+=1;continue
                row=choice(dict(tools=tools,messages=r['messages']),'Which response should the assistant give next, given the available tools?',options,'preferred',family,origin,group)
            # Anonymous keys prevent a preferred/rejected key from revealing the answer.
            oldkeys=list(row['question']['criteria']); oldgold=row['gold']
            row['question']['criteria']={f'option_{i}':row['question']['criteria'][k] for i,k in enumerate(oldkeys)}
            row['gold']=f'option_{oldkeys.index(oldgold)}'
            row['source']='v5:'+family
            row['provenance']['content_keys']=sorted(keys)
            for key in keys:components.union(group,key)
            candidates.append(row)
    forced={components.find(r['provenance']['group']) for r in candidates if set(r['provenance']['content_keys'])&old_atoms}
    tok=AutoTokenizer.from_pretrained('mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8',local_files_only=True)
    bins=defaultdict(list);seen=set()
    # Random traversal also bounds tokenization cost after ample eligible groups.
    random.Random(SEED).shuffle(candidates)
    for r in candidates:
        p=r['provenance'];group=components.find(p['group']);n=int(digest([SEED,group])[:8],16)/2**32
        split='train' if group in forced or n<.75 else 'val' if n<.84 else 'calibration' if n<.92 else 'test'
        signature=statekey(r['state'])
        if signature in seen:counts['duplicate_state']+=1;continue
        seen.add(signature)
        if len(bins[split,r['family']])>5000:continue
        if len(tok.encode(render_state(r['state']),add_special_tokens=False))>2048:counts['long_full_state']+=1;continue
        if len(tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False))>3000:counts['long_full_prompt']+=1;continue
        p.update(component=group,split=split);bins[split,r['family']].append(r)
    splits=defaultdict(list)
    for (split,family),rs in sorted(bins.items()):
        random.Random(SEED).shuffle(rs)
        rs=rs[:2000 if split=='train' else 150]
        assert len(rs)>=(150 if split!='train' else 1500),(split,family,len(rs))
        splits[split]+=rs
    owners={};origins=set()
    for split,rs in splits.items():
        for r in rs:
            assert r['provenance']['origin'] not in origins;origins.add(r['provenance']['origin'])
            for key in [r['provenance']['component'],statekey(r['state'])]+r['provenance']['content_keys']:
                assert owners.setdefault(key,split)==split,(key,split)
            assert not(set(r['provenance']['content_keys'])&blocked)
            assert r['target'][list(r['question']['criteria']).index(r['gold'])]==1
    def balanced(rs,limit):
        groups=defaultdict(list)
        for r in rs:groups[r.get('family',r['source'])].append(r)
        for k,g in sorted(groups.items()):random.Random(SEED).shuffle(g)
        result=[]
        while groups and len(result)<limit:
            for k in list(groups):
                result.append(groups[k].pop())
                if not groups[k]:del groups[k]
                if len(result)==limit:break
        return result
    retention=balanced(parent,8000)
    payload={names[0]:retention+splits['train'],names[1]:balanced(read(data/'selection_v4_tools.jsonl'),800)+splits['val'],names[2]:balanced(read(data/'calibration_v4_tools.jsonl'),800)+splits['calibration'],names[3]:splits['test']}
    for name,rs in payload.items():random.Random(SEED).shuffle(rs);write(data/name,rs)
    smoke=sorted(payload[names[0]],key=lambda r:len(build_prompt(tok,r['state'],Question.from_dict(r['question']))),reverse=True)[:16]+balanced(payload[names[0]],32)
    write(work/'smoke.jsonl',smoke)
    report=dict(seed=SEED,sources=[dict(s,sha256=sha(s['path'])) for s in specs],counts={name:dict(Counter(r.get('family',r['source']) for r in rs)) for name,rs in payload.items()},exclusions=dict(counts),files={n:sha(data/n) for n in names},protected={str(p):sha(p) for p in protected},builder_sha256=sha(__file__),notes=['Evaluation partitions used only for exclusions.','New holdouts use source training partitions; no native test labels used for selection.','Full current Decision Index 0.2 frozen corpus unavailable; no official overall score.','Related video/article/premise/tool-catalog/query components stay together.','Exact normalized text exclusion does not detect all paraphrases.'])
    (data/'train_v5.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':report['counts'],'exclusions':report['exclusions']},indent=2))
if __name__=='__main__':main()
