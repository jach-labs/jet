"""Target weak task families using training partitions only; retain broad skills."""
import json,hashlib,random,sys
from pathlib import Path
from collections import Counter,defaultdict
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from format import Question,build_prompt
from data.decision_training import atoms,statekey,normalized,digest
HERE=Path(__file__).resolve().parent;SEED=24092417
rng=random.Random(SEED)
def read(p):
 with Path(p).open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def keys(r):
 state=r.get('state',{})
 k=set(atoms(state))
 if state:k.add(statekey(state))
 k.update(x.removeprefix('content:') for x in r.get('provenance',{}).get('content_keys',[]))
 return k

def group(r):return str(r.get('provenance',{}).get('group') or statekey(r['state']))
def category(r):
 s=r['source']
 if 'code_' in s:return 'code'
 if 'stance' in s:return 'stance'
 if 'sarcasm' in s:return 'sarcasm'
 if 'product_relevance' in s:return 'relevance'
 if 'preference_transfer' in s:return 'preference'
 return 'retention'

blocked=set();blocked_groups=set();protected=[]
for p in (ROOT/'data').glob('*.jsonl'):
 if any(x in p.name for x in ['val','test','selection','calibration']):
  protected.append(p)
  for r in read(p):blocked.update(keys(r));blocked_groups.add(group(r))
bench=ROOT/'experiments/jet-4b-full-index/expanded.jsonl';protected.append(bench)
for r in read(bench):
 blocked.update(keys(r))
 for q in r['questions'].values():
  # Long exact instruction identities catch prompts whose benchmark state is empty.
  text=q.get('instructions','')
  if len(text)>=80:blocked.add(digest(normalized(text)))

pool={};removed=Counter()
for n in ['train_v4.jsonl','train_v5_r2.jsonl']:
 for r in read(ROOT/'data'/n):
  if keys(r)&blocked or group(r) in blocked_groups:removed['protected_overlap']+=1;continue
  pool.setdefault(digest([r['state'],r['question']]),r)
tok=AutoTokenizer.from_pretrained(ROOT/'releases/jet-v6')
def fits(r):
 ids=tok.encode(build_prompt(tok,r['state'],Question.from_dict(r['question'])),add_special_tokens=False)
 return len(ids)<=3072
buckets=defaultdict(list)
for r in pool.values():buckets[category(r)].append(r)
train=[]
for kind,cap in [('code',5000),('stance',5000),('sarcasm',2000),('relevance',2000),('retention',6000)]:
 rows=buckets[kind];rng.shuffle(rows)
 if kind=='retention':
  bysource=defaultdict(list)
  for r in rows:bysource[r['source']].append(r)
  rows=[]
  while any(bysource.values()):
   for source in sorted(bysource):
    if bysource[source]:rows.append(bysource[source].pop())
 n=0
 for r in rows:
  if n>=cap:break
  if not fits(r):removed['too_long']+=1;continue
  train.append(r);n+=1

spec=json.loads((ROOT/'work/jet-targeted-20260924/preference-source.json').read_text())
newtrain=[];newval=[];seen=set()
for batch in pq.ParquetFile(spec['file']).iter_batches(batch_size=256):
 for r in batch.to_pylist():
  prompt=r['prompt'];identity=statekey(prompt)
  if identity in seen:continue
  seen.add(identity)
  row={'state':prompt,'source':'targeted:preference_transfer','family':'preference','provenance':{'repo':spec['repo'],'revision':spec['revision'],'original_split':'train_prefs','group':identity,'label_source':'published synthetic preference pairs'}}
  if keys(row)&blocked:removed['preference_protected_overlap']+=1;continue
  chosen=r['chosen'][-1]['content'];rejected=r['rejected'][-1]['content']
  if chosen==rejected:continue
  options=[(chosen,1.),(rejected,0.)];random.Random(SEED+int(identity[:12],16)).shuffle(options)
  row['question']={'type':'choice','instructions':'Which response better follows the request and is more helpful, accurate, and appropriate?','criteria':{'A':options[0][0],'B':options[1][0]}}
  row['target']=[o[1] for o in options]
  row['gold']='A' if options[0][1] else 'B'
  val=int(identity[:8],16)%20==0
  if len(newval if val else newtrain)>=(150 if val else 3000):continue
  if not fits(row):removed['preference_too_long']+=1;continue
  (newval if val else newtrain).append(row)
 if len(newtrain)==3000 and len(newval)==150:break
assert len(newtrain)==3000 and len(newval)==150
train+=newtrain;rng.shuffle(train)
val=list(read(ROOT/'data/selection_v5.jsonl'))+newval
assert not ({group(r) for r in train}&{group(r) for r in val})
assert not ({statekey(r['state']) for r in train}&{statekey(r['state']) for r in val})
for name,rows in [('train',train),('selection',val)]:
 p=HERE/(name+'.jsonl');assert not p.exists()
 p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
audit={'seed':SEED,'train_rows':len(train),'selection_rows':len(val),'train_categories':dict(Counter(category(r) for r in train)),'train_sources':dict(Counter(r['source'] for r in train)),'filters':dict(removed),'new_preference_train':3000,'new_preference_selection':150,'benchmark_rows_used_as_training':0,'protected_inputs':[str(p.relative_to(ROOT)) for p in protected],'source':spec,'source_sha256':hashlib.file_digest(open(spec['file'],'rb'),'sha256').hexdigest(),'files':{n:hashlib.file_digest(open(HERE/n,'rb'),'sha256').hexdigest() for n in ['train.jsonl','selection.jsonl']},'limitations':'Exact state/content/group exclusions, not semantic decontamination. Existing model training exposure cannot be undone. Preference-transfer labels are synthetic, not new human consensus labels. Benchmark results have informed task-family selection.'}
(HERE/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
