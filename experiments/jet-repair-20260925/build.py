"""Repair task selection using unexposed source groups; never train on benchmark answers."""
import csv,hashlib,json,random,sys
from pathlib import Path
from collections import Counter,defaultdict
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from data.decision_training import atoms,statekey,normalized,digest,choice
from format import Question,build_prompt
from transformers import AutoTokenizer
SEED=250925;rng=random.Random(SEED)
def read(p):
 for l in p.open():
  if l.strip():yield json.loads(l)
def keys(r):return set(atoms(r['state']))|{statekey(r['state'])}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
oldtrain=list((ROOT/'data').glob('train*.jsonl'))+[ROOT/'experiments/jet-targeted-20260924/train.jsonl',ROOT/'experiments/jet-4b-comparison/train.jsonl']
protected=[p for p in (ROOT/'data').glob('*.jsonl') if p not in oldtrain]+[ROOT/'experiments/jet-targeted-20260924/selection.jsonl',ROOT/'experiments/jet-4b-comparison/validation.jsonl',ROOT/'experiments/jet-4b-full-index/expanded.jsonl']
seen=set();blocked=set();oldtopics=set();seengroups=set();blockedgroups=set()
for p in oldtrain:
 for r in read(p):
  seen.update(keys(r));seengroups.add(r.get('provenance',{}).get('group'))
  if 'stance' in r.get('source','') and isinstance(r['state'],dict):oldtopics.add(normalized(r['state'].get('target','')))
for p in protected:
 for r in read(p):blocked.update(keys(r));blockedgroups.add(r.get('provenance',{}).get('group'))
seen|=blocked;seengroups|=blockedgroups;seengroups.discard(None);blockedgroups.discard(None)
train=[];val=[];test=[];audit={};sources=[]
tok=AutoTokenizer.from_pretrained(ROOT/'releases/jet-v6')
def fits(r):return len(tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False))<=3072
def make(state,options,gold,family,origin,group):
 r=choice(state,{'sarcasm':'Is this text intended to be sarcastic?','stance':'What stance does the author express towards the target?','code':'What does f(**arguments) return?','relevance':'How relevant is the product to the search query?','preference':'Which response better follows the request and is more helpful, accurate, and appropriate?'}[family],options,gold,family,origin,group)
 r['source']='repair:'+family
 # Neutral option identifiers with independently shuffled option order.
 original=list(r['question']['criteria']);order=list(range(len(original)));random.Random(digest([SEED,origin])).shuffle(order)
 r['question']['criteria']={f'option_{j}':r['question']['criteria'][original[i]] for j,i in enumerate(order)}
 r['target']=[r['target'][i] for i in order];r['gold']=f'option_{r["target"].index(1.)}'
 r['provenance']['semantic_gold']=gold
 return r
# Sarcasm: grouped author tweets/rephrases, official train only; new holdout tweets.
p=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw/repos/isarcasm/train/train.En.csv';sources.append({'file':str(p),'sha256':sha(p),'revision':'dfc708b53bde1bb571abfb5692f63231c2232195'})
groups=[]
for i,r in enumerate(csv.DictReader(p.open())):
 if not r.get('tweet'):continue
 rs=[make(r['tweet'],{'yes':'sarcastic','no':'not sarcastic'},'yes' if int(r['sarcastic']) else 'no','sarcasm',f'isarcasm:{i}',f'isarcasm:{i}')]
 if r.get('rephrase'):rs.append(make(r['rephrase'],{'yes':'sarcastic','no':'not sarcastic'},'no','sarcasm',f'isarcasm:{i}:rephrase',f'isarcasm:{i}'))
 if any(keys(x)&blocked for x in rs):continue
 groups.append(rs)
rng.shuffle(groups);held=set()
for split,dest,counts in [('selection',val,{'yes':75,'no':225}),('test',test,{'yes':50,'no':150})]:
 got=Counter()
 for rs in groups:
  g=rs[0]['provenance']['group'];label=rs[0]['provenance']['semantic_gold']
  if g in held or got[label]>=counts[label] or any(keys(x)&seen for x in rs):continue
  dest.append(rs[0]);held.add(g);got[label]+=1
 assert dict(got)==counts,(split,got)
 audit['sarcasm_'+split]=dict(got)
for label,cap in [('yes',375),('no',1125)]:
 candidates=[r for rs in groups if rs[0]['provenance']['group'] not in held for r in rs if r['provenance']['semantic_gold']==label]
 rng.shuffle(candidates);assert len(candidates)>=cap;train+=candidates[:cap]
# VAST: new selection/test topics absent from every previous local training set.
p=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw/vast/data/VAST/vast_train.csv';sources.append({'file':str(p),'sha256':sha(p),'revision':'e7c4775182b184730f350995f8260579c9e066fe'})
rows=[]
for r in csv.DictReader(p.open()):
 row=make({'text':r['post'],'target':r['new_topic']},{'against':'opposes the target','favor':'supports the target','none':'no clear stance towards the target'},['against','favor','none'][int(r['label'])],'stance','vast:'+r['new_id'],'vast-article:'+r['arc_id'])
 if not keys(row)&blocked and row['provenance']['group'] not in blockedgroups and fits(row):rows.append(row)
rng.shuffle(rows);topics={};heldkeys=set();heldgroups=set()
for dest,split,cap in [(val,'selection',200),(test,'test',150)]:
 selected=[]
 for row in rows:
  topic=normalized(row['state']['target']);g=row['provenance']['group']
  if g in seengroups or topic in oldtopics or (topic in topics and topics[topic]!=split) or keys(row)&seen or keys(row)&heldkeys or g in heldgroups:continue
  topics[topic]=split;selected.append(row)
  if len(selected)==cap:break
 assert len(selected)==cap,(split,len(selected))
 dest+=selected
 for r in selected:heldkeys|=keys(r);heldgroups.add(r['provenance']['group'])
stance=[r for r in rows if normalized(r['state']['target']) not in topics and not keys(r)&heldkeys and r['provenance']['group'] not in heldgroups]
train+=stance[:1500];assert len(stance)>=1500
audit['stance_held_topics']={s:sum(v==s for v in topics.values()) for s in ['selection','test']}
# New executable programs: complete templates kept apart, including from old pools.
bodies=[
 'return [v + x.count(v) for v in x if v >= k]',
 'return sum((i + 1) * v for i, v in enumerate(x) if v < k)',
 'return [a - b for a, b in zip(x, x[1:]) if a + b >= k]',
 'return sorted(set(x))[::-1][k:]',
 'return [sum(x[i:i+2]) for i in range(len(x)-1) if i % 2 == k % 2]',
 'return sum(1 for a in x for b in x if a * b > k)',
 'return [v for i, v in enumerate(x) if v not in x[:i] and v != k]',
 'return sum(v * v for v in x) - k * len(set(x))',
 'return [max(x[:i+1]) - min(x[:i+1]) + k for i in range(len(x))]',
 'return [v - k if i % 2 else v + k for i, v in enumerate(sorted(x))]',
 'return sum(x[i] * x[i+1] for i in range(len(x)-1)) + k',
 'return [len([v for v in x if v < t]) for t in x[k:]]',
 'return sorted([v for v in x if x.count(v) == 1], reverse=True)[:k+1]',
 'return sum(abs(a-b) for a,b in zip(x,x[1:])) - k',
]
safe={n:__builtins__[n] if isinstance(__builtins__,dict) else getattr(__builtins__,n) for n in ['sum','len','max','min','sorted','set','enumerate','zip','range','abs']}
for tid,body in enumerate(bodies):
 split='train' if tid<8 else 'selection' if tid<11 else 'test';dest={'train':train,'selection':val,'test':test}[split];cap=125 if split=='train' else 100 if split=='selection' else 67
 code='def f(x, k):\n    '+body;ns={};exec(code,{'__builtins__':{},**safe},ns);fn=ns['f'];unique=set()
 for attempt in range(10000):
  x=[rng.randint(-8,9) for _ in range(rng.randint(4,10))];k=rng.randint(0,3);y=fn(x,k)
  alts=[fn(x[::-1],k),fn(x,k+1),fn(x[1:],k)]
  alts += [y+[k],y+[k+1],y+[k+2]] if isinstance(y,list) else [y-1,y+1,y+2]
  answers=list(dict.fromkeys(map(repr,[y,*alts])))[:4]
  if len(answers)<4:continue
  r=make({'code':code,'arguments':{'x':x,'k':k}},{str(i):v for i,v in enumerate(answers)},'0','code',f'newcode:{tid}:{attempt}',f'newcode-template:{tid}')
  if keys(r)&seen or statekey(r['state']) in unique:continue
  unique.add(statekey(r['state']));r['provenance']['oracle']='Python execution of local fixed template';dest.append(r)
  if len(unique)==cap:break
 assert len(unique)==cap
# Generic preference remains a proxy; unseen prompt groups supply larger holdouts.
spec=json.loads((ROOT/'work/jet-targeted-20260924/preference-source.json').read_text());p=Path(spec['file']);sources.append({**spec,'sha256':sha(p)})
counts=Counter();groupset=set()
for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
 for r in batch.to_pylist():
  g=statekey(r['prompt'])
  if g in groupset:continue
  groupset.add(g)
  row=make(r['prompt'],{'chosen':r['chosen'][-1]['content'],'rejected':r['rejected'][-1]['content']},'chosen','preference','uf:'+g,'uf:'+g)
  if keys(row)&seen or not fits(row) or r['chosen'][-1]['content']==r['rejected'][-1]['content']:continue
  split='selection' if counts['selection']<200 else 'test' if counts['test']<200 else 'train'
  if counts['train']>=750:continue
  {'train':train,'selection':val,'test':test}[split].append(row);counts[split]+=1
 if counts['train']>=750:break
assert counts=={'selection':200,'test':200,'train':750},counts
# ESCI fresh query groups, held product/content protected across splits.
spec=next(s for s in json.loads((ROOT/'data/train_v3.provenance.json').read_text())['sources'] if s['repo']=='tasksource/esci');sources.append(spec)
from data.decision_training import public_candidates
rows=[];unique=set()
for row in public_candidates([spec]):
 if row['family']!='product_relevance':continue
 row['source']='repair:relevance'
 sig=statekey(row['state'])
 if sig in unique or keys(row)&seen or row['provenance']['group'] in seengroups or not fits(row):continue
 unique.add(sig);rows.append(row)
rng.shuffle(rows);assigned={};crosskeys=set();counts=Counter()
for row in rows:
 g=row['provenance']['group'];split=assigned.get(g)
 target='selection' if counts['selection']<200 else 'test' if counts['test']<200 else 'train'
 if split and split!=target:continue
 if keys(row)&crosskeys:continue
 assigned[g]=target;{'selection':val,'test':test,'train':train}[target].append(row);counts[target]+=1;crosskeys|=keys(row)
 if counts['train']==750:break
assert counts=={'selection':200,'test':200,'train':750},counts
# Retention samples stay broad. Retention validation is existing, not called fresh.
prior=list(read(ROOT/'experiments/jet-targeted-20260924/train.jsonl'))
focus=['code','stance','sarcasm','relevance','preference']
retention=[r for r in prior if not any(k in r['source'] for k in focus)];rng.shuffle(retention)
heldall=set().union(*(keys(r) for r in val+test))
retention=[r for r in retention if not keys(r)&heldall];train+=retention[:2500];assert len(retention)>=2500
oldval=list(read(ROOT/'experiments/jet-targeted-20260924/selection.jsonl'))
val += [r for r in oldval if not any(k in r['source'] for k in focus)]
# Reserve evaluation content even when different source IDs repeat the same text.
testkeys=set().union(*(keys(r) for r in test));testgroups={r['provenance']['group'] for r in test}
val=[r for r in val if not keys(r)&testkeys and r.get('provenance',{}).get('group') not in testgroups]
heldkeys=set().union(*(keys(r) for r in val+test));heldgroups={r.get('provenance',{}).get('group') for r in val+test};heldgroups.discard(None)
before=len(train);train=[r for r in train if not keys(r)&heldkeys and r.get('provenance',{}).get('group') not in heldgroups]
audit['final_cross_source_training_exclusions']=before-len(train)
assert len(train)>=7500,len(train)
# Enforce content/group separation and fresh focus selection/test versus all prior data.
for name,rows in [('train',train),('selection',val),('test',test)]:
 for r in rows:assert fits(r),(name,r['source'])
 for other,others in [('train',train),('selection',val),('test',test)]:
  if name>=other:continue
  assert not (set().union(*(keys(r) for r in rows)) & set().union(*(keys(r) for r in others))),(name,other,'content')
  assert not ({r.get('provenance',{}).get('group') for r in rows if r.get('provenance',{}).get('group')} & {r.get('provenance',{}).get('group') for r in others if r.get('provenance',{}).get('group')}),(name,other,'groups')
 for r in rows:
  if name!='train' and r['source'].startswith('repair:'):
   assert not keys(r)&seen
   assert r['provenance']['group'] not in seengroups
for name,rows in [('train',train),('selection',val),('test',test)]:
 rng.shuffle(rows);p=HERE/(name+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 audit[name]={'n':len(rows),'sources':dict(Counter(r['source'] for r in rows)),'sha256':sha(p)}
audit.update(seed=SEED,sources=sources,prior_training_files=[str(p) for p in oldtrain],protected_files=[str(p) for p in protected],limitations=['Exact content and group exclusions are not semantic decontamination or a pretraining contamination guarantee.','New focus holdouts are unused by previous local runs; retention validation is reused.','Generic preference examples remain a proxy for consensus; no claim of fresh Habermas testing.','Code templates are disjoint but share Python primitives.','Benchmark confusion counts informed this development run; benchmark cases never enter training.'])
(HERE/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['train','selection','test','sarcasm_selection','sarcasm_test','stance_held_topics']},indent=2))
