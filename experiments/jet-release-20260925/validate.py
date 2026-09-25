import gc,hashlib,json,sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'releases/jet-v6.1'
sys.path.insert(0,str(ROOT/'src'))
from qwen35_training import load_model,label_logits
from format import Question,label_token_ids
from inference import encode
from decision_index_engine import as_text
mode=sys.argv[1];torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
if mode=='candidate':
 rows=[];counts=Counter()
 for p in [ROOT/'data/test_v5_new.jsonl',ROOT/'data/calibration_v5.jsonl']:
  for line in p.open():
   row=json.loads(line);kind=row['question']['type']
   if counts[kind]<32:rows.append(row);counts[kind]+=1
 assert counts=={'choice':32,'score':32,'noul':32}
 # Two fixed supported requests per dataset; choose by hash, not correctness.
 rp=ROOT/'artifacts/decision-index/runs/jet-kev-comparison-20260925/results.jsonl'
 supported={r['run_id'] for r in map(json.loads,rp.open()) if r['status']=='ok'}
 chosen={}
 for r in map(json.loads,(ROOT/'experiments/jet-kev-comparison-20260925/rows.jsonl').open()):
  e=r['_evaluation'];bid=e['catalog_id']
  if e['run_id'] not in supported:continue
  order=hashlib.sha256(e['run_id'].encode()).hexdigest();entries=chosen.setdefault(bid,[]);entries.append((order,r));entries.sort(key=lambda x:x[0]);del entries[2:]
 for entries in chosen.values():
  for _,r in entries:
   q=dict(next(iter(r['questions'].values())));q['instructions']=as_text(q['instructions']);q['criteria']={k:as_text(v) for k,v in q['criteria'].items()};rows.append({'state':r['state'],'question':q})
 (HERE/'merge-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 model,tok,_=load_model(str(ROOT/'releases/jet-v6'),'e5b8f610ddb92ffaba596ae452bed32a9fef49ca',adapter=str(ROOT/'adapters/jet-targeted-20260924/best'))
else:
 sys.path.insert(0,str(OUT));from jet import Jet
 wrapper=Jet(str(OUT));model,tok=wrapper.model,wrapper.tokenizer
 rows=list(map(json.loads,(HERE/'merge-cases.jsonl').open()))
refs=[];temp=json.loads((OUT/'calibration.json').read_text())
with torch.no_grad():
 for row in rows:
  q=Question.from_dict(row['question']);ids=encode(tok,row['state'],q,10**9);assert len(ids)<=8192
  z=label_logits(model,{'ids':ids,'labels':label_token_ids(tok,q)}).double();prob=(z/temp[q.type]).softmax(-1).cpu().tolist();refs.append({'type':q.type,'probabilities':prob})
if mode=='candidate':(HERE/'candidate-probabilities.json').write_text(json.dumps(refs)+'\n')
else:
 prior=json.loads((HERE/'candidate-probabilities.json').read_text());records=[]
 for i,(a,b) in enumerate(zip(prior,refs)):
  ap=a['probabilities'];bp=b['probabilities'];records.append({'case':i,'type':a['type'],'same_argmax':max(range(len(ap)),key=ap.__getitem__)==max(range(len(bp)),key=bp.__getitem__),'max_probability_difference':max(abs(x-y) for x,y in zip(ap,bp))})
 smoke=wrapper.decide('The item arrived broken.',{'category':{'type':'choice','instructions':'What happened?','criteria':{'damage':'The item was damaged.','billing':'A payment problem.'}},'severity':{'type':'score','instructions':'Rate severity.','criteria':['none','moderate','severe']},'damaged':{'type':'noul','instructions':'Is the item damaged?'}})
 assert set(smoke['answers'])=={'category','severity','damaged'}
 result={'cases':len(records),'argmax_flips':sum(not r['same_argmax'] for r in records),'max_probability_difference':max(r['max_probability_difference'] for r in records),'selection':'First 32 local cases per type plus two hash-selected supported benchmark requests per dataset, first question only. Not an independent performance evaluation.','calibration':'Inherited v6 temperatures, not refitted for this continuation.','base_download_required':False,'wrapper_smoke':smoke,'records':records}
 result['passed']=result['argmax_flips']/result['cases']<=.03 and result['max_probability_difference']<.075
 (OUT/'merge-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','wrapper_smoke']}),flush=True)
 assert result['passed'],'Merge diverged; inspect before publishing'
