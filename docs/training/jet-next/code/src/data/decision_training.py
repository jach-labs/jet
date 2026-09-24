"""Reproducible v3 data from pinned TRAIN files and locally executable generators.

Run after scripts/rebuild_index_diagnostic.py and scripts/download_v3_sources.py:
  uv run --no-sync python -m data.decision_training
No benchmark normalizer supplies training rows. Evaluation is exclusion-only.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from format import Question, render_state, build_prompt

SEED = 230923
ROOT = Path(__file__).resolve().parents[2]

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def normalized(s):
    return ' '.join(s.casefold().split())

def atoms(value):
    if isinstance(value, str):
        if len(normalized(value)) >= 32:
            yield digest(normalized(value))
    elif isinstance(value, dict):
        for v in value.values(): yield from atoms(v)
    elif isinstance(value, list):
        for v in value: yield from atoms(v)

def statekey(value):
    return digest(normalized(value) if isinstance(value, str) else value)

def choice(state, instruction, options, gold, family, origin, group, identities=(), oracle=None):
    rng = random.Random(SEED + int(digest([origin, family])[:12], 16))
    keys = list(options); rng.shuffle(keys)
    if family in ('arithmetic', 'code_behavior'):
        # Anonymous keys must not reveal the right answer (e.g. output_0).
        gold_position = keys.index(gold)
        options = {f'option_{i}': options[k] for i, k in enumerate(keys)}
        keys = list(options)
        gold = keys[gold_position]
    result = dict(state=state, question=dict(type='choice', instructions=instruction,
        criteria={k: options[k] for k in keys}), target=[float(k == gold) for k in keys],
        source='v3:'+family, family=family, provenance=dict(origin=origin, original_split='train',
        group=group, identities=list(identities)), gold=gold)
    if oracle is not None: result['provenance']['oracle'] = oracle
    return result

def public_candidates(sources):
    for spec in sources:
        repo, file = spec['repo'], spec['file']
        if 'glaive' in repo:
            yield from tool_candidates(spec)
            continue
        table = pq.read_table(spec['path'])
        indices = list(range(len(table))); random.Random(SEED).shuffle(indices)
        # Bounded deterministic subset, not a new partition of the public evaluation files.
        for idx in indices[:40000]:
            r = table.slice(idx, 1).to_pylist()[0]
            origin = f'{repo}@{spec["revision"]}/{file}:{r.get("example_id", r.get("idx",idx))}'
            if 'esci' in repo:
                if r['product_locale'] != 'us': continue
                product = {k: r[k] for k in ['product_title','product_description','product_bullet_point','product_brand','product_color'] if r[k]}
                if len(json.dumps(product)) > 5000: continue
                options = dict(Exact='exact match to the requested product', Substitute='a substitute that can satisfy the same need', Complement='an accessory or complementary product', Irrelevant='irrelevant to the search intent')
                yield choice(dict(search_query=r['query'],product=product),'How relevant is the product to the search query?',options,r['esci_label'],'product_relevance',origin,f'esci-query:{r["query_id"]}',[f'esci-product:{r["product_id"]}', 'query:'+normalized(r['query'])])
            elif 'snli' in repo:
                if r['label'] < 0: continue
                options=dict(entailment='the premise implies the hypothesis',neutral='the premise does not determine whether the hypothesis is true',contradiction='the premise contradicts the hypothesis')
                yield choice(dict(premise=r['premise'],hypothesis=r['hypothesis']),'What is the relation of the hypothesis to the premise?',options,list(options)[r['label']],'entailment',origin,'snli-premise:'+normalized(r['premise']))
            elif 'glue' in repo:
                yield choice(dict(query=r['question'],document=r['sentence']),'Does this document sentence contain the answer to the query?',dict(yes='contains the answer',no='does not contain the answer'),'yes' if r['label']==0 else 'no','document_relevance',origin,'qnli-query:'+normalized(r['question']),['qnli-doc:'+normalized(r['sentence'])])
            elif 'irony' in file:
                yield choice(r['text'],'Does the text express verbal irony?',dict(yes='ironic',no='not ironic'),'yes' if r['label'] else 'no','sarcasm_irony',origin,'tweet:'+normalized(r['text']))
            elif 'stance' in file:
                topic={'stance_abortion':'legalization of abortion','stance_atheism':'atheism','stance_climate':'climate change is a real concern','stance_feminist':'the feminist movement','stance_hillary':'Hillary Clinton'}[file.split('/')[0]]
                options=dict(none='no clear stance towards the target',against='opposes the target',favor='supports the target')
                yield choice(dict(text=r['text'],target=topic),'What stance does the author express towards the target?',options,list(options)[r['label']],'stance',origin,'tweet:'+normalized(r['text']))
    # Actual author-labelled sarcasm, separate official TRAIN partition. Rephrases stay grouped.
    file=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw/repos/isarcasm/train/train.En.csv'
    for idx,r in enumerate(csv.DictReader(file.open())):
        if not r.get('tweet'): continue
        origin=f'iSarcasmEval@dfc708b53bde1bb571abfb5692f63231c2232195/train/train.En.csv:{idx}'
        group='isarcasm:'+str(idx)
        yield choice(r['tweet'],'Is the author being sarcastic?',dict(yes='sarcastic',no='not sarcastic'),'yes' if int(r['sarcastic']) else 'no','sarcasm_irony',origin,group)
        if r.get('rephrase'):
            yield choice(r['rephrase'],'Is the author being sarcastic?',dict(yes='sarcastic',no='not sarcastic'),'no','sarcasm_irony',origin+':rephrase',group)

def tool_candidates(spec):
    records=json.load(open(spec['path']))
    parsed=[]
    relevance=[]
    for idx,r in enumerate(records):
        # Use only the first request and its immediate labelled function call.
        match=re.search(r'USER:\s*(.*?)\n+ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|\n\n)',r['chat'],re.S)
        if not match: continue
        try:
            positive=match[2].startswith('<functioncall>')
            call=ast.literal_eval(match[2].replace('<functioncall>', '', 1).strip()) if positive else None
            tools=[]; rem=r['system'][r['system'].index('{'):]
            while rem.strip():
                item,end=json.JSONDecoder().raw_decode(rem.lstrip());tools.append(item);rem=rem.lstrip()[end:]
            names=[x['name'] for x in tools]
            if len(set(names)) != len(names): continue
            if positive and call['name'] not in names: continue
            if not positive and not re.search(r"(don't|do not) have (the )?(capability|function|ability)|only (have|allows|function)|cannot (assist|help|perform)", match[2], re.I): continue
            if len(match[1]) > 1800: continue
        except (ValueError,SyntaxError,KeyError,TypeError): continue
        if len(tools)==1:
            relevance.append((idx,match[1].strip(),tools,positive))
        if positive: parsed.append((idx,match[1].strip(),tools,call['name']))
    catalog={t['name']:t['description'] for _,_,ts,_ in parsed for t in ts}
    names=sorted(catalog)
    for idx,query,tools,gold in parsed:
        origin=f'{spec["repo"]}@{spec["revision"]}/{spec["file"]}:{idx}'
        group='tool-catalog:'+digest(sorted(t['name'] for t in tools))
        rng=random.Random(SEED+idx)
        options={t['name']:t['description'] for t in tools}
        # Selection uses source-labelled calls plus catalog distractors. Binary relevance
        # uses the native single-tool call/refusal annotation, without added distractors.
        others=[n for n in names if n not in options]
        for n in rng.sample(others,min(5,len(others))): options[n]=catalog[n]
        if sum(len(x) for x in options.values())>2200: continue
        yield choice(query,'Select the tool that directly fulfills the user request.',options,gold,'tool_selection',origin,group)
    for idx,query,tools,positive in relevance:
        tool=tools[0]
        origin=f'{spec["repo"]}@{spec["revision"]}/{spec["file"]}:{idx}'
        group='tool-catalog:'+digest([tool['name']])
        yield choice(dict(request=query,tool=tool),'Is this tool relevant for fulfilling the user request?',
            dict(yes='the tool can fulfill the request',no='the request requires a different capability'),
            'yes' if positive else 'no','tool_relevance',origin,group)

def generators(n=10000):
    rng=random.Random(SEED)
    for i in range(n):
        values={x:bool(rng.getrandbits(1)) for x in ['active','verified','paid','blocked','admin']}
        def rule(depth):
            if depth==0 or rng.random()<0.22:
                name=rng.choice(list(values));return f'not {name}' if rng.random()<0.3 else name
            return f'({rule(depth-1)} {rng.choice(["and","or"])} {rule(depth-1)})'
        expr=rule(rng.randint(2,4)); result=bool(eval(expr,{'__builtins__':{}},values))
        origin=f'generator:boolean:{i}'; group='boolean-rule:'+expr
        yield choice(dict(facts=values,allow_if=expr),'Under this exact Boolean rule, is access allowed?',dict(allow='access is allowed',deny='access is denied'),'allow' if result else 'deny','boolean_rules',origin,group,oracle=dict(expression=expr,variables=values,result=result))
        a,b,c=[rng.randint(-30,50) for _ in range(3)];d=rng.randint(1,12)
        expr=rng.choice([f'({a} + {b}) * {c}',f'{a} * {b} - {c}',f'({a} - {b}) // {d}',f'{a} + {b} * {c}',f'({a} * {b}) % {d}',f'abs({a} - {b}) + {c}'])
        result=eval(expr,{'__builtins__':{},'abs':abs})
        choices={result,result+1,result-1,result+rng.choice([3,5,10])}
        options={f'value_{j}':str(v) for j,v in enumerate(sorted(choices))};gold=next(k for k,v in options.items() if v==str(result))
        yield choice(dict(expression=expr),'Evaluate the Python integer expression. // is floor division and % is modulo.',options,gold,'arithmetic',f'generator:arithmetic:{i}','arithmetic:'+expr,oracle=dict(expression=expr,result=result))
        k=rng.randint(-8,12);m=rng.randint(2,8);inp=[rng.randint(-10,15) for _ in range(rng.randint(3,7))]
        body=rng.choice([f'return sum(v + {k} for v in x if v % {m} != 0)',f'return [v * {k} for v in x][::{m}]',f'return sorted(set(x + [{k}]))[:{m}]',f'return sum(x[-{m}:]) + {k}',f'return [v for v in x if v > {k}][::-1]',f'return len([v for v in x if v % {m} == {k%m}])'])
        code='def f(x):\n    '+body
        ns={};exec(code,{'__builtins__':{},'sum':sum,'sorted':sorted,'set':set,'len':len},ns)
        result=ns['f'](inp)
        if isinstance(result,list): bad=[result[::-1],result+[k],result[1:],[]]
        else: bad=[result-1,result+1,result+3,0]
        outputs=list(dict.fromkeys([repr(result)]+[repr(x) for x in bad]))[:4]
        if len(outputs)<2: continue
        options={f'output_{j}':v for j,v in enumerate(outputs)}
        yield choice(dict(code=code,input=inp),'What value does f return for the supplied input?',options,'output_0','code_behavior',f'generator:code:{i}','program:'+code,oracle=dict(result=result,executable=True))

class Components:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while x!=self.parent[x]:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def read(path):return [json.loads(line) for line in path.open() if line.strip()]
def write(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def main():
    sources=json.load(open(ROOT/'work/jet-v3/sources.json'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('mlx-community/Qwen3-0.6B-bf16')
    data=ROOT/'data'; protected=[data/x for x in ['val.jsonl','test.jsonl','score_eval.jsonl']]
    suite=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl'
    assert suite.exists(), 'Rebuild the evaluation corpus first, for exclusion-only use.'
    held=[r for p in protected+[suite] for r in read(p)]
    blocked_states={statekey(r['state']) for r in held}
    blocked_atoms={a for r in held for a in atoms(r['state'])}
    legacy=read(data/'train_v2.jsonl')
    legacy_keys={statekey(r['state']) for r in legacy}
    legacy_atoms={a for r in legacy for a in atoms(r['state'])}
    stats=Counter();seen=set();candidates=[]
    for r in list(public_candidates(sources))+list(generators()):
        if statekey(r['state']) in blocked_states or set(atoms(r['state'])) & blocked_atoms:
            stats['excluded_evaluation_content']+=1;continue
        # Original identity + family permits multiple tasks from one source row, kept together.
        identity=(r['provenance']['origin'],r['family'])
        semantic=digest([r['state'],r['question']['instructions'],r['question']['criteria']])
        if identity in seen or semantic in seen:
            stats['duplicate']+=1;continue
        if len(tokenizer.encode(render_state(r['state']), add_special_tokens=False)) > 1024:
            stats['over_state_capacity']+=1;continue
        if len(tokenizer.encode(build_prompt(tokenizer,r['state'],Question.from_dict(r['question'])), add_special_tokens=False)) > 2048:
            stats['over_prompt_capacity']+=1;continue
        seen.add(identity);seen.add(semantic);candidates.append(r)
    components=Components()
    for r in candidates:
        p=r['provenance'];key=p['group']
        links=['origin:'+p['origin'],'state:'+statekey(r['state'])]+p['identities']
        if r['family'] in ('tool_selection', 'tool_relevance'):
            query=r['state'] if isinstance(r['state'],str) else r['state']['request']
            links.append('tool-query:'+normalized(query))
        # Entire same tweet/premise/query/product/program components stay in one partition.
        for link in links:components.union(key,link)
    forced_train=set()
    for r in candidates:
        if statekey(r['state']) in legacy_keys or set(atoms(r['state'])) & legacy_atoms:
            forced_train.add(components.find(r['provenance']['group']))
    bins=defaultdict(list)
    for r in candidates:
        group=components.find(r['provenance']['group']);b=int(digest([SEED,group])[:8],16)/2**32
        split='train' if group in forced_train or b<.76 else 'val' if b<.84 else 'calibration' if b<.92 else 'test'
        r['provenance']['component']=group;r['provenance']['split']=split
        bins[(split,r['family'])].append(r)
    out=defaultdict(list);counts={}
    for (split,family),rows in sorted(bins.items()):
        # Equal family budget, with round-robin classes; no duplicate oversampling.
        random.Random(SEED).shuffle(rows);classes=defaultdict(list)
        for r in rows:classes[r['gold'] if family not in ['tool_selection','arithmetic','code_behavior'] else 'all'].append(r)
        selected=[];limit=2000 if split=='train' else 160
        while classes and len(selected)<limit:
            for key in list(classes):
                selected.append(classes[key].pop())
                if not classes[key]:del classes[key]
                if len(selected)==limit:break
        out[split].extend(selected);counts[split+':'+family]=len(selected)
    # Remove legacy content colliding with known protected evaluation; never alter train_v2.
    clean_legacy=[];seen_legacy=set()
    for r in legacy:
        if statekey(r['state']) in blocked_states:
            stats['legacy_evaluation_overlap']+=1;continue
        signature=digest([r['state'],r['question']])
        if signature in seen_legacy:stats['legacy_duplicate']+=1;continue
        seen_legacy.add(signature);clean_legacy.append(r)
    out['train']=clean_legacy+out['train']
    # Split existing val by state into checkpoint-selection and calibration (file untouched).
    original_val=read(data/'val.jsonl')
    selection=[r for r in original_val if int(statekey(r['state'])[:8],16)%2==0]
    calibration=[r for r in original_val if int(statekey(r['state'])[:8],16)%2!=0]
    files={'train_v3.jsonl':out['train'],'val_v3_new.jsonl':out['val'],
        'calibration_v3_new.jsonl':out['calibration'],'test_v3_new.jsonl':out['test'],
        'selection_v3.jsonl':selection+out['val'],'calibration_v3.jsonl':calibration+out['calibration']}
    hashes={}
    for name,rows in files.items():
        random.Random(SEED).shuffle(rows)
        for r in rows:
            Question.from_dict(r['question']); assert abs(sum(r['target'])-1)<1e-6
        write(data/name,rows);hashes[name]=hashlib.sha256((data/name).read_bytes()).hexdigest()
    # A group and original source ID may occur many times, but only in one split.
    owners={}
    for split,rows in out.items():
        for r in rows:
            if 'provenance' not in r:continue
            for key in [r['provenance']['component'],r['provenance']['origin'],statekey(r['state'])]:
                assert owners.setdefault(key,split)==split, (key,split)
    provenance=dict(seed=SEED,sources=[{**s,'sha256':hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()} for s in sources],
        isarcasm=dict(repository='https://github.com/iabufarha/iSarcasmEval',revision='dfc708b53bde1bb571abfb5692f63231c2232195',file='train/train.En.csv',split='train'),
        protected_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected+[suite]},
        counts=counts,filters=dict(stats),files=hashes,legacy_rows=len(clean_legacy),generator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Official frozen suite unavailable; exclusions cover all 23,113 rebuilt requests and existing held-outs.', 'ESCI uses only the first pinned training shard; official ESCI evaluation uses test.', 'New held-outs are grouped samples of training partitions, not leaderboard evaluation.', 'Glaive labels are public synthetic annotations; added distractors are not exhaustively verified for semantic equivalence.', 'Legacy train_v2 lacks original source IDs; retained legacy provenance is its immutable file hash.'])
    (data/'train_v3.provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(counts=counts,filters=dict(stats),total_train=len(out['train'])),indent=2))

if __name__=='__main__':main()
