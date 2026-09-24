"""Add training-partition stance/contracts and executable multi-step Python tasks.

Never uses evaluation labels. Existing files are read-only exclusion/retention inputs.
New holdouts are grouped by article/document or whole program template.
"""
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from data.decision_training import ROOT, atoms, choice, digest, normalized, read, statekey, write
from format import Question, render_state, build_prompt
from transformers import AutoTokenizer

SEED = 230924
RAW = ROOT / 'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw'
# Distinct complete templates are assigned to each split before generating inputs.
PROGRAMS = [
    'out = []\nfor v in x:\n    if v > k:\n        out.append(v * 2)\nreturn out',
    'out = 0\nfor i, v in enumerate(x):\n    out += v if i % 2 == 0 else -v\nreturn out + k',
    'out = []\nfor v in x:\n    if v not in out:\n        out.append(v)\nreturn out[k:]',
    'out = []\nfor i in range(1, len(x)):\n    out.append(x[i] - x[i-1])\nreturn out[:k]',
    'out = 1\nfor v in x:\n    if v != 0:\n        out *= v\nreturn out + k',
    'out = []\nfor v in x:\n    if v >= k:\n        out.insert(0, v)\nreturn out',
    'out = 0\nfor v in x:\n    if v == k:\n        break\n    out += v\nreturn out',
    'out = []\nfor v in x:\n    if v % 2:\n        continue\n    out.append(v + k)\nreturn out',
    'out = []\nfor a, b in zip(x, x[1:]):\n    if a < b:\n        out.append(a + b + k)\nreturn out',
    'out = []\nfor v in x:\n    out.append(sum(x[:len(out)+1]))\nreturn out[k:]',
    'out = sorted(x)\nif k % 2:\n    out.reverse()\nreturn out[:k]',
    'out = []\nfor v in x:\n    if x.count(v) > k and v not in out:\n        out.append(v)\nreturn out',
    'out = 0\nfor v in reversed(x):\n    out = out * 2 + v\nreturn out - k',
    'out = []\nfor i, v in enumerate(x):\n    if v > sum(x[:i]):\n        out.append(i)\nreturn out[k:]',
    'out = []\nfor v in x:\n    if out and out[-1] == v:\n        out.pop()\n    else:\n        out.append(v)\nreturn out + [k]',
    'out = 0\nfor a in x:\n    for b in x:\n        if a + b == k:\n            out += 1\nreturn out',
    'out = list(x)\nfor i in range(k):\n    out = out[1:] + out[:1]\nreturn out',
    'out = []\nfor v in x:\n    out.append(v)\n    if v == k:\n        out.append(-v)\nreturn out',
]

def program_rows():
    safe = dict(enumerate=enumerate, range=range, len=len, zip=zip, sum=sum,
                sorted=sorted, reversed=reversed, list=list)
    for tid, body in enumerate(PROGRAMS):
        split = 'train' if tid < 12 else 'val' if tid < 14 else 'calibration' if tid < 16 else 'test'
        code = 'def f(x, k):\n' + '\n'.join('    '+line for line in body.splitlines())
        ns = {}; exec(code, {'__builtins__': {}, **safe}, ns); fn = ns['f']
        rng = random.Random(SEED + tid)
        for i in range(300 if split == 'train' else 120):
            x = [rng.randint(-4, 7) for _ in range(rng.randint(3, 8))]; k = rng.randint(0, 4)
            result = fn(x.copy(), k)
            # Counterfactual inputs/parameters produce plausible same-type distractors.
            alternatives = [fn(x.copy(), k+1), fn(x[::-1], k), fn(x[1:], k), fn(x+[k], k)]
            alternatives += ([result+[k], result+[k+1], result+[k+2]] if isinstance(result, list)
                             else [result-1, result+1, result+2])
            vals = list(dict.fromkeys([repr(result)] + [repr(v) for v in alternatives]))[:4]
            options = {f'candidate_{j}': v for j,v in enumerate(vals)}
            r = choice({'code':code,'arguments':{'x':x,'k':k}}, 'What does f(**arguments) return?',
                       options,'candidate_0','code_behavior',f'local-v4:{tid}:{i}',f'v4-template:{tid}')
            # choice randomizes semantic keys again for this family.
            r['family']='code_transfer'; r['source']='v4:code_transfer'
            r['provenance'].update(split=split, oracle={'result':result,'template':tid}, original_split='generated')
            yield r

def main():
    data=ROOT/'data'; outputs=['train_v4.jsonl','selection_v4.jsonl','calibration_v4.jsonl','test_v4_new.jsonl']
    if any((data/p).exists() for p in outputs): raise SystemExit('Refusing to overwrite existing v4 files')
    tokenizer=AutoTokenizer.from_pretrained('mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8')
    old_train=read(data/'train_v3.jsonl')
    protected=[data/p for p in ['val.jsonl','test.jsonl','score_eval.jsonl','val_v3_new.jsonl','calibration_v3_new.jsonl','test_v3_new.jsonl']]
    protected += [ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl']
    held_atoms={a for p in protected for r in read(p) for a in atoms(r['state'])}
    held_states={statekey(r['state']) for p in protected for r in read(p)}
    seen={digest([r['state'],r['question']]) for r in old_train}
    old_atoms={a for r in old_train for a in atoms(r['state'])}
    sources=[]; candidates=list(program_rows()); excluded=Counter()
    vast=RAW/'vast/data/VAST/vast_train.csv'
    sources.append(dict(repository='https://github.com/emilyallaway/zero-shot-stance',revision='e7c4775182b184730f350995f8260579c9e066fe',file='data/VAST/vast_train.csv',sha256=hashlib.sha256(vast.read_bytes()).hexdigest(),split='train'))
    # Native development/test posts are exclusion-only, including all topic variants.
    blocked_posts={normalized(r['post']) for f in ['vast_dev.csv','vast_test.csv'] for r in csv.DictReader((vast.parent/f).open())}
    for r in csv.DictReader(vast.open()):
        if normalized(r['post']) in blocked_posts: excluded['vast_native_heldout_post']+=1; continue
        row=choice({'text':r['post'],'target':r['new_topic']},'What stance does the author express towards the target?',
            dict(against='opposes the target',favor='supports the target',none='no clear stance towards the target'),
            ['against','favor','none'][int(r['label'])],'stance_transfer','vast-train:'+r['new_id'],'vast-article:'+r['arc_id'])
        candidates.append(row)
    archive=RAW/'repos/contractnli/resources/contract-nli.zip'
    sources.append(dict(repository='https://github.com/stanfordnlp/contract-nli',revision='eced6528dd3c1d14d73f9a87df8f7bdbc03126f9',file='resources/contract-nli.zip!contract-nli/train.json',sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),split='train'))
    with zipfile.ZipFile(archive) as z:
        contracts=json.loads(z.read('contract-nli/train.json'))
        held_contracts={normalized(d['text']) for s in ['dev','test'] for d in json.loads(z.read(f'contract-nli/{s}.json'))['documents']}
    for d in contracts['documents']:
        if normalized(d['text']) in held_contracts: excluded['contract_native_heldout_text']+=1; continue
        # Keep the full contract. Never assign a full-document label to a cropped context.
        for lid,a in d['annotation_sets'][0]['annotations'].items():
            row=choice({'contract':d['text'],'hypothesis':contracts['labels'][lid]['hypothesis']},
                'Does the full contract entail, contradict, or leave the hypothesis undetermined?',
                dict(Entailment='entailed by the contract',Contradiction='contradicted by the contract',NotMentioned='not determined by the contract'),
                a['choice'],'contract_entailment',f'contractnli-train:{d["id"]}:{lid}',f'contractnli-document:{d["id"]}')
            candidates.append(row)
    # Same normalized document/post stays grouped even if source IDs differ.
    from data.decision_training import Components
    components=Components()
    for r in candidates:
        group=r['provenance']['group']; s=r['state']
        text=s.get('contract',s.get('text'))
        if text: components.union(group,'document:'+digest(normalized(text)))
    forced_train=set()
    for r in candidates:
        s=r['state']; text=s.get('contract',s.get('text'))
        if text and digest(normalized(text)) in old_atoms:
            forced_train.add(components.find(r['provenance']['group']))
    bins=defaultdict(list)
    for r in candidates:
        p=r['provenance'];s=r['state'];text=s.get('contract',s.get('text'))
        # Global hypotheses recur by design; only document content is an exclusion key.
        if statekey(s) in held_states or (text and digest(normalized(text)) in held_atoms):
            excluded['existing_evaluation_content']+=1;continue
        signature=digest([s,r['question']])
        if signature in seen: excluded['duplicate']+=1;continue
        seen.add(signature)
        if len(tokenizer.encode(render_state(s),add_special_tokens=False))>2048:
            excluded['over_2048_full_state']+=1;continue
        if len(tokenizer.encode(build_prompt(tokenizer,s,Question.from_dict(r['question'])),add_special_tokens=False))>2304:
            excluded['over_2304_full_prompt']+=1;continue
        group=components.find(p['group']); b=int(digest([SEED,group])[:8],16)/2**32
        split=p.get('split') or ('train' if group in forced_train or b<.8 else 'val' if b<.87 else 'calibration' if b<.94 else 'test')
        p.update(component=group,split=split);r['source']='v4:'+r['family'];bins[(split,r['family'])].append(r)
    splits=defaultdict(list);counts={}
    for (split,family),rows in sorted(bins.items()):
        random.Random(SEED).shuffle(rows)
        classes=defaultdict(list)
        for r in rows:classes[r['gold'] if family!='code_transfer' else 'all'].append(r)
        selected=[];limit=3000 if split=='train' else 240
        while classes and len(selected)<limit:
            for key in list(classes):
                selected.append(classes[key].pop())
                if not classes[key]:del classes[key]
                if len(selected)==limit:break
        splits[split].extend(selected);counts[split+':'+family]=len(selected)
    owners={}
    for split,rows in splits.items():
        for r in rows:
            for key in [r['provenance']['component'],r['provenance']['origin'],statekey(r['state'])]:
                assert owners.setdefault(key,split)==split,(key,split)
    payload={outputs[0]:old_train+splits['train'],outputs[1]:read(data/'selection_v3.jsonl')+splits['val'],
             outputs[2]:read(data/'calibration_v3.jsonl')+splits['calibration'],outputs[3]:splits['test']}
    class_weights={}
    for family in ['contract_entailment','stance_transfer']:
        rows=[r for r in splits['train'] if r['family']==family]
        counts_by_class=Counter(r['gold'] for r in rows)
        raw={k:min(3.0,len(rows)/(len(counts_by_class)*n)) for k,n in counts_by_class.items()}
        mean=sum(raw[k]*n for k,n in counts_by_class.items())/len(rows)
        weights={k:w/mean for k,w in raw.items()};class_weights[family]=weights
        for r in rows:r['weight']=weights[r['gold']]
    hashes={}
    for name,rows in payload.items():
        random.Random(SEED).shuffle(rows);write(data/name,rows);hashes[name]=hashlib.sha256((data/name).read_bytes()).hexdigest()
    report=dict(seed=SEED,sources=sources,counts=counts,excluded=dict(excluded),files=hashes,class_weights=class_weights,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        retained_train_v3_sha256=hashlib.sha256((data/'train_v3.jsonl').read_bytes()).hexdigest(),
        protected_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
        limitations=['Full frozen suite remains unavailable.', 'ContractNLI restricted to complete contracts fitting 2048 state tokens.',
        'Source article/document groups are held together. Code holdouts use disjoint complete templates, but share Python primitives.',
        'Exact normalized content exclusion is not a semantic paraphrase detector.', 'V3 tool tests remain small and easy; do not interpret their accuracy as strong tool retrieval evidence.'])
    (data/'train_v4.provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__': main()
