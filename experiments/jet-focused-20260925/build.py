"""Freeze source-group splits before training; benchmark inputs are exclusion-only."""
import ast,csv,hashlib,json,random,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from data.decision_training import choice,digest,normalized
from format import Question,build_prompt
SEED=260925;rng=random.Random(SEED);WORK=ROOT/'work'/HERE.name

def read(p):
 for l in p.open():
  if l.strip():yield json.loads(l)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strings(x):
 if isinstance(x,str):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from strings(v)
 elif isinstance(x,list):
  for v in x:yield from strings(v)
def keys(r):
 # Include embedded requests from benchmark instructions, not just state.
 values=list(strings(r.get('state',{})))
 for q in r.get('questions',{}).values():
  ins=q.get('instructions','')
  if isinstance(ins,str):values += [ins,*ins.split('\n')]
 return {digest(normalized(v)) for v in values if len(normalized(v))>=20}
def group(r):return r.get('provenance',{}).get('group')
def make(state,instruction,options,gold,family,origin,g):
 r=choice(state,instruction,options,gold,family,origin,g);r['source']='focused:'+family
 old=list(r['question']['criteria']);order=list(range(len(old)));random.Random(digest([SEED,origin])).shuffle(order)
 r['question']['criteria']={f'option_{j}':r['question']['criteria'][old[i]] for j,i in enumerate(order)}
 r['target']=[r['target'][i] for i in order];r['gold']=f'option_{r["target"].index(1.)}';return r

def main():
 audit={'seed':SEED,'sources':json.loads((WORK/'sources.json').read_text())};seen=set();blocked=set();seen_groups=set();blocked_groups=set();prior=[]
 paths=list((ROOT/'data').glob('*.jsonl'))
 for p in (ROOT/'experiments').glob('**/*.jsonl'):
  if p.parent==HERE:continue
  if p.name in ['train.jsonl','selection.jsonl','test.jsonl','validation.jsonl','expanded.jsonl','rows.jsonl','merge-cases.jsonl']:paths.append(p)
 for p in sorted(set(paths)):
  istrain=p.name.startswith('train');prior.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'training':istrain})
  for r in read(p):
   ks=keys(r);seen.update(ks)
   if group(r):seen_groups.add(group(r))
   if not istrain:
    blocked.update(ks)
    if group(r):blocked_groups.add(group(r))
 audit['prior_inputs']=prior;print('Historical input files:',len(prior),flush=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'releases/jet-v6.1')
 def fits(r):return len(tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False))<=3072
 pools={};bank=pq.read_table(WORK/'banking-train.parquet').to_pylist();labels=sorted({r['label_text'] for r in bank});assert len(labels)==77
 pools['banking']=[[make(r['text'],'Classify the banking intent of this user request.',{x:x for x in labels},r['label_text'],'banking',f'banking-train:{i}','banking:'+digest(normalized(r['text'])))] for i,r in enumerate(bank)]
 finance=[]
 for i,r in enumerate(csv.DictReader((WORK/'SEntFiN.csv').open())):
  title=r['Title']
  try:entities=ast.literal_eval(r['Decisions'])
  except (SyntaxError,ValueError):
   pairs=re.findall(r"'(.+?)': '(negative|neutral|positive)'(?=, |})",r['Decisions'])
   assert '{'+', '.join("'"+k+"': '"+v+"'" for k,v in pairs)+'}'==r['Decisions'],r['Decisions']
   entities=dict(pairs)
  rows=[]
  for ent,label in entities.items():
   assert label.lower() in ['negative','neutral','positive']
   rows.append(make({'document':title,'entity':ent},'Classify the sentiment toward the supplied financial entity. Consider only that entity, even when the document mentions others.',{x:x for x in ['Negative','Neutral','Positive']},label.title(),'finance',f'sentfin:{i}:{ent}','sentfin:'+digest(normalized(title))))
  finance.append(rows)
 pools['finance']=finance
 p=ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/raw/repos/isarcasm/train/train.En.csv'
 audit['sources'].append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'revision':'dfc708b53bde1bb571abfb5692f63231c2232195'})
 sarcasm=[]
 for i,r in enumerate(csv.DictReader(p.open())):
  if not r.get('tweet'):continue
  rows=[make(r['tweet'],'Is this text intended to be sarcastic?',{'yes':'sarcastic','no':'not sarcastic'},'yes' if int(r['sarcastic']) else 'no','sarcasm',f'isarcasm:{i}',f'isarcasm:{i}')]
  if r.get('rephrase'):rows.append(make(r['rephrase'],'Is this text intended to be sarcastic?',{'yes':'sarcastic','no':'not sarcastic'},'no','sarcasm',f'isarcasm:{i}:rephrase',f'isarcasm:{i}'))
  sarcasm.append(rows)
 pools['sarcasm']=sarcasm
 split={'train':[],'selection':[],'test':[]};held_keys=set();held_groups=set();audit['available']={}
 for family,groups in pools.items():
  # Keep all entities/rephrases of a document in one partition.
  groups=[rs for rs in groups if all(not keys(r)&blocked and group(r) not in blocked_groups and fits(r) for r in rs)]
  rng.shuffle(groups);fresh=[rs for rs in groups if all(not keys(r)&seen and group(r) not in seen_groups for r in rs)]
  audit['available'][family]={'groups':len(groups),'fresh_groups':len(fresh)};print(family,audit['available'][family],flush=True)
  for name in ['selection','test']:
   counts=Counter();target=154 if family=='banking' else 180 if family=='finance' else 80
   for rs in fresh:
    if any(group(r) in held_groups or keys(r)&held_keys for r in rs):continue
    chosen=rs if family=='finance' else rs[:1]
    if family=='banking':
     label=next(k for k,v in chosen[0]['question']['criteria'].items() if chosen[0]['target'][list(chosen[0]['question']['criteria']).index(k)]==1);semantic=chosen[0]['question']['criteria'][label]
     if counts[semantic]>=2:continue
    elif family=='sarcasm':
     semantic=list(chosen[0]['question']['criteria'].values())[chosen[0]['target'].index(1.)]
     if counts[semantic]>= (20 if semantic=='sarcastic' else 60):continue
    else:semantic='all'
    split[name]+=chosen;counts[semantic]+=len(chosen)
    for r in rs:held_keys|=keys(r);held_groups.add(group(r))
    if sum(counts.values())>=target:break
   assert sum(counts.values())>=target,(family,name,counts,audit['available'][family])
   audit[f'{family}_{name}']=dict(counts)
  candidates=[r for rs in groups if all(group(x) not in held_groups and not keys(x)&held_keys for x in rs) for r in rs]
  rng.shuffle(candidates);chosen=[];dedup=set();counts=Counter();cap=800 if family!='sarcasm' else 400
  # Prioritize genuine multi-entity finance headlines; preserve every sentiment class.
  if family=='finance':
   multi={group(r) for rs in groups if len(rs)>1 for r in rs};candidates.sort(key=lambda r:group(r) not in multi)
  for r in candidates:
   k=digest([r['state'],r['question']]);label=list(r['question']['criteria'].values())[r['target'].index(1.)]
   if k in dedup:continue
   if family=='sarcasm' and counts[label]>=(100 if label=='sarcastic' else 300):continue
   if family=='banking' and counts[label]>=11:continue
   if family=='finance' and counts[label]>=267:continue
   chosen.append(r);dedup.add(k);counts[label]+=1
   if len(chosen)==cap:break
  assert len(chosen)==cap,(family,len(chosen),counts);split['train']+=chosen
 # Broad replay from previously vetted training pool; deterministic round-robin by source.
 retention=defaultdict(list)
 for r in read(ROOT/'data/train_v5.jsonl'):
  if any(x in r['source'] for x in ['banking','sarcasm','irony']):continue
  if keys(r)&(blocked|held_keys) or group(r) in blocked_groups|held_groups or not fits(r):continue
  r['source']='retention:'+r['source'];retention[r['source']].append(r)
 for rows in retention.values():rng.shuffle(rows)
 dedup=set();replay=[]
 while len(replay)<2000:
  progress=False
  for source in sorted(retention):
   while retention[source]:
    r=retention[source].pop();k=digest(r['state'])
    if k not in dedup:break
   else:continue
   replay.append(r);dedup.add(k);progress=True
   if len(replay)==2000:break
  assert progress
 split['train']+=replay
 # Retention validation/test are explicitly reused historical holdouts.
 for name,path in [('selection',ROOT/'data/selection_v5.jsonl'),('test',ROOT/'data/test_v5_new.jsonl')]:
  other_rows=[r for other,rs in split.items() if other!=name for r in rs]
  other_keys=set().union(*(keys(r) for r in other_rows));other_groups={group(r) for r in other_rows if group(r)}
  rows=[r for r in read(path) if not any(x in r['source'] for x in ['banking','sarcasm','irony']) and not keys(r)&other_keys and group(r) not in other_groups and fits(r)];rng.shuffle(rows)
  assert len(rows)>=500,(name,'retention',len(rows))
  for r in rows[:500]:r['source']='retention:'+r['source']
  split[name]+=rows[:500]
 # Drop any replay row overlapping a new focus row in another partition.
 for a in split:
  for b in split:
   if a>=b:continue
   ka=set().union(*(keys(r) for r in split[a]));kb=set().union(*(keys(r) for r in split[b]));assert not ka&kb,(a,b,'content overlap')
   ga={group(r) for r in split[a] if group(r)};gb={group(r) for r in split[b] if group(r)};assert not ga&gb,(a,b,'group overlap')
 for name,rows in split.items():
  rng.shuffle(rows);p=HERE/(name+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  audit[name]={'n':len(rows),'sources':dict(Counter(r['source'] for r in rows)),'sha256':sha(p),'max_tokens':max(len(tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False)) for r in rows)}
 audit['limitations']=['Exact content/group exclusions do not establish semantic or pretraining decontamination.','Focus holdouts exclude previous local inputs; retention holdouts are reused.','SEntFiN measures entity-level sentiment transfer, not FinEntity benchmark performance.','Benchmarks informed target selection; final benchmark reruns are development evaluation.','No synthetic paraphrases or benchmark test labels are used as training supervision.']
 (HERE/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['available','train','selection','test']},indent=2))
if __name__=='__main__':main()
