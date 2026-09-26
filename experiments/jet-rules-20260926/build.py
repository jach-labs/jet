import hashlib,json,random,sys
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from generate import rules,deadlines,TREES,SEED
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from format import Question,build_prompt
rng=random.Random(SEED)
def read(p):return [json.loads(l) for l in p.open() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def statekey(r):return hashlib.sha256(json.dumps(r['state'],sort_keys=True,ensure_ascii=False).casefold().encode()).hexdigest()
def contents(x):
 if isinstance(x,str):
  s=' '.join(x.casefold().split())
  if len(s)>30:yield hashlib.sha256(s.encode()).hexdigest()
 elif isinstance(x,dict):
  for k,v in x.items():
   if k not in ['policy','instructions']:yield from contents(v)
 elif isinstance(x,list):
  for v in x:yield from contents(v)
def main():
 tok=AutoTokenizer.from_pretrained(ROOT/'releases/jet-v6.2');splits={};inputs={}
 def fits(r):return len(tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False))<=3072
 for split,n in [('train',1000),('selection',160),('test',160)]:
  splits[split]=[r for family in TREES for r in rules(split,family,n)]+deadlines(split,n)
 for split in ['selection','test']:
  p=ROOT/'experiments/jet-focused-20260925'/(split+'.jsonl');inputs[str(p.relative_to(ROOT))]=sha(p);splits[split]+=read(p)
 suite=ROOT/'experiments/jet-kev-family-20260926/suite.jsonl';bench=read(suite);inputs[str(suite.relative_to(ROOT))]=sha(suite)
 protected=splits['selection']+splits['test']+bench;blocked_states={statekey(r) for r in protected};blocked_text=set().union(*(set(contents(r['state'])) for r in protected))
 # Preserve previous protected partitions in broad replay; never sample old heldout rows.
 for p in (ROOT/'data').glob('*.jsonl'):
  if p.name.startswith('train'):continue
  inputs[str(p.relative_to(ROOT))]=sha(p)
  for r in read(p):blocked_states.add(statekey(r));blocked_text.update(contents(r['state']))
 def allowed(r):return statekey(r) not in blocked_states and not set(contents(r['state']))&blocked_text and fits(r)
 p=ROOT/'data/train_v5.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p);groups=defaultdict(list)
 for r in read(p):
  if allowed(r):r['source']='retention:'+r['source'];groups[r['source']].append(r)
 for rs in groups.values():rng.shuffle(rs)
 replay=[];used=set()
 while len(replay)<1000:
  progress=False
  for source in sorted(groups):
   while groups[source]:
    r=groups[source].pop();k=statekey(r)
    if k not in used:break
   else:continue
   used.add(k);replay.append(r);progress=True
   if len(replay)==1000:break
  assert progress
 p=ROOT/'experiments/jet-focused-20260925/train.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p);old=read(p);rng.shuffle(old)
 for family,cap in [('banking',302),('finance',300),('sarcasm',398)]:
  rows=[r for r in old if r['source']=='focused:'+family and allowed(r) and statekey(r) not in used];assert len(rows)>=cap,(family,len(rows));replay+=rows[:cap];used.update(statekey(r) for r in rows[:cap])
 splits['train']+=replay;assert len(splits['train'])==6000
 # Exact full-case protection, plus explicit disjoint AST signatures and date-year ranges.
 owners={};source_counts={}
 for split,rows in splits.items():
  for r in rows:
   key=statekey(r);assert owners.setdefault(key,split)==split,(split,'case overlap')
   assert fits(r)
  rng.shuffle(rows);p=HERE/(split+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));source_counts[split]={'n':len(rows),'sha256':sha(p),'sources':dict(Counter(r['source'] for r in rows))}
 assert not ({statekey(r) for r in splits['train']} & {statekey(r) for r in bench})
 audit={'seed':SEED,'inputs':inputs,**source_counts,'rule_trees':TREES,'date_years':{'train':[2018,2031],'selection':[2033,2037],'test':[2038,2042]},'limitations':['This is benchmark-informed development; targeted rule families are now included in training.','New synthetic holdouts use disjoint complete rule ASTs and date-year ranges; they share primitives and some linguistic conventions.','Banking, finance, sarcasm and broad retention holdouts are reused.','Exact-case/content screening does not establish semantic or pretraining decontamination.','Kev transfer-v4 is consulted once after checkpoint selection; its measured cases and gold labels are not used to generate supervision.']}
 (HERE/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(source_counts,indent=2))
if __name__=='__main__':main()
