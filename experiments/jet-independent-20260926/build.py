"""Freeze new examples before evaluating one already-selected checkpoint."""
import ast,calendar,csv,datetime as dt,hashlib,importlib.util,itertools,json,random,re,sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
b=module('focused_build',ROOT/'experiments/jet-focused-20260925/build.py');g=module('rules_generator',ROOT/'experiments/jet-rules-20260926/generate.py')
SEED=26092617;rng=random.Random(SEED);g.SEED=SEED
NEW={
 'and_or':[('and',('or',0,('not',1)),('or',2,3)),('and',('or',0,1),('or',2,('not',3)))],
 'or_not':[('or',('and',0,('not',1)),('not',('and',2,3))),('or',('and',('not',0),1),('not',('or',2,3)))],
 'conditional':[('if',('or',0,1),('and',2,3),('not',2)),('if',('and',0,('not',1)),('not',2),('or',2,3))]}
for fam,trees in NEW.items():
 prior=[t for split in g.TREES[fam].values() for t in split];assert not set(map(repr,trees))&set(map(repr,prior))
 for tree in trees:
  for bits in itertools.product([False,True],repeat=4):assert g.evaluate(tree,bits)==eval(g.expression(tree),{'__builtins__':{}},{'b':bits})
 g.TREES[fam]['independent']=trees

def main():
 assert not (HERE/'fresh.jsonl').exists()
 paths=list((ROOT/'data').glob('*.jsonl'))
 names={'train.jsonl','selection.jsonl','test.jsonl','validation.jsonl','expanded.jsonl','rows.jsonl','merge-cases.jsonl','suite.jsonl','fresh.jsonl'}
 paths += [p for p in (ROOT/'experiments').glob('**/*.jsonl') if p.parent!=HERE and p.name in names]
 seen=set();groups=set();states=set();inputs={}
 for p in sorted(set(paths)):
  inputs[str(p.relative_to(ROOT))]=b.sha(p)
  for row in b.read(p):
   seen.update(b.keys(row));states.add(b.digest(row.get('state')))
   if b.group(row):groups.add(b.group(row))
 print('Exclusion files',len(inputs),flush=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'releases/jet-v6.2')
 def fresh(rs):return all(not b.keys(r)&seen and b.group(r) not in groups and b.digest(r['state']) not in states for r in rs)
 def fits(r):return len(tok.encode(b.build_prompt(tok,r['state'],b.Question.from_dict(r['question'])),add_special_tokens=False))<=3072
 rows=[]
 for fam in NEW:
  rs=g.rules('independent',fam,400);assert fresh(rs);rows+=rs
 for i in range(402):
  y=rng.randint(2045,2055);m=rng.randint(1,12);d=rng.choice([1,calendar.monthrange(y,m)[1],rng.randint(1,28)])
  due=dt.date(y,m,d);grace=rng.choice([1,2,3,7,10,14,21,30]);label=i%3
  offset=rng.choice([-7,-1,0]) if label==0 else rng.choice([1,grace]) if label==1 else rng.choice([grace+1,grace+2,grace+14])
  received=due+dt.timedelta(days=offset);fmt=['%Y-%m-%d','%B %d, %Y','%d %B %Y'][(i//3)%3]
  state={'policy':f'Classify arrival as on time when received by the due date (inclusive), late but accepted during the next {grace} calendar days (inclusive), and rejected after that.','case':{'due date':due.strftime(fmt),'receipt date':received.strftime(fmt)}}
  assert g.deadline_label(due,received,grace)==label
  row=g.decision(state,'score',label,'deadline',f'independent:deadline:{i}',{'deadline':due.isoformat(),'received':received.isoformat(),'grace_days':grace,'days_after_deadline':offset,'label':label},rng);assert fresh([row]);rows.append(row)
 pools={};raw=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw/repos/isarcasm/train/train.En.csv';sources={str(raw.relative_to(ROOT)):b.sha(raw)}
 pools['sarcasm']=[]
 for i,r in enumerate(csv.DictReader(raw.open())):
  if not r.get('tweet'):continue
  rs=[b.make(r['tweet'],'Is this text intended to be sarcastic?',{'yes':'sarcastic','no':'not sarcastic'},'yes' if int(r['sarcastic']) else 'no','sarcasm',f'isarcasm:{i}',f'isarcasm:{i}')]
  if r.get('rephrase'):rs.append(b.make(r['rephrase'],'Is this text intended to be sarcastic?',{'yes':'sarcastic','no':'not sarcastic'},'no','sarcasm',f'isarcasm:{i}:rephrase',f'isarcasm:{i}'))
  if fresh(rs):pools['sarcasm'].append(rs[0])
 raw=ROOT/'work/jet-focused-20260925/banking-train.parquet';sources[str(raw.relative_to(ROOT))]=b.sha(raw);bank=pq.read_table(raw).to_pylist();labels=sorted({r['label_text'] for r in bank});pools['banking']=[]
 for i,r in enumerate(bank):
  row=b.make(r['text'],'Classify the banking intent of this user request.',{x:x for x in labels},r['label_text'],'banking',f'banking-train:{i}','banking:'+b.digest(b.normalized(r['text'])))
  if fresh([row]):pools['banking'].append(row)
 raw=ROOT/'work/jet-focused-20260925/SEntFiN.csv';sources[str(raw.relative_to(ROOT))]=b.sha(raw);pools['finance']=[]
 for i,r in enumerate(csv.DictReader(raw.open())):
  try:entities=ast.literal_eval(r['Decisions'])
  except (SyntaxError,ValueError):
   pairs=re.findall(r"'(.+?)': '(negative|neutral|positive)'(?=, |})",r['Decisions']);assert '{'+', '.join("'"+k+"': '"+v+"'" for k,v in pairs)+'}'==r['Decisions'];entities=dict(pairs)
  rs=[b.make({'document':r['Title'],'entity':ent},'Classify the sentiment toward the supplied financial entity. Consider only that entity, even when the document mentions others.',{x:x for x in ['Negative','Neutral','Positive']},label.title(),'finance',f'sentfin:{i}:{ent}','sentfin:'+b.digest(b.normalized(r['Title']))) for ent,label in entities.items()]
  if fresh(rs):pools['finance']+=rs
 selected_keys=set();selected_groups=set();counts={}
 for fam,pool in pools.items():
  rng.shuffle(pool);c=Counter()
  for row in pool:
   label=list(row['question']['criteria'].values())[row['target'].index(1.)];cap=(80 if label=='sarcastic' else 240) if fam=='sarcasm' else 60 if fam=='finance' else 2
   if c[label]>=cap or b.keys(row)&selected_keys or b.group(row) in selected_groups or not fits(row):continue
   rows.append(row);c[label]+=1;selected_keys.update(b.keys(row));selected_groups.add(b.group(row))
  counts[fam]=dict(c);print(fam,dict(c),flush=True)
  assert sum(c.values())>=({'sarcasm':20,'banking':100,'finance':100}[fam]),(fam,c)
 assert all(fits(r) for r in rows)
 assert len({b.digest(r['state']) for r in rows})==len(rows)
 rng.shuffle(rows);path=HERE/'fresh.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 audit={'seed':SEED,'n':len(rows),'sha256':b.sha(path),'families':dict(Counter(r['source'] for r in rows)),'class_counts':counts,'exclusions':inputs,'raw_sources':sources,'new_trees':NEW,'limitations':['Independent examples relative to known local training/evaluation artifacts; cannot exclude unknown original backbone pretraining.','Synthetic tasks share the generator primitives and domains of prior training; this measures new compositions and examples, not entirely new domains.','Public source examples use the official training partitions but are excluded from all known local training and evaluation.','Broad retention is measured separately on the reused diagnostic split.','Only 6 unused sarcastic examples remain under the strict local exclusions; fresh sarcasm F1 is exploratory and insufficient to establish preservation.']}
 (HERE/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
