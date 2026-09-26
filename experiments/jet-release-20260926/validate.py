"""Check fixed merge cases and all frozen holdout metrics before publication."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'releases/jet-v6.2';FOCUS=ROOT/'experiments/jet-focused-20260925'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(FOCUS))
from train import read_data,evaluate,eligible
from qwen35_training import load_model,label_logits
from format import Question,label_token_ids
from inference import encode
from decision_index_engine import as_text
mode=sys.argv[1];assert mode in ['candidate','merged']
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
if mode=='candidate':
 rows=list(map(json.loads,(ROOT/'experiments/jet-release-20260925/merge-cases.jsonl').open()))
 assert len(rows)==144
 for fam in ['banking','finance','sarcasm']:
  candidates=[r for r in map(json.loads,(FOCUS/'test.jsonl').open()) if r['source']=='focused:'+fam]
  candidates.sort(key=lambda r:hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest());rows+=candidates[:4]
 r=json.loads((FOCUS/'api-bank-longest.jsonl').read_text());q=dict(next(iter(r['questions'].values())));q['instructions']=as_text(q['instructions']);q['criteria']={k:as_text(v) for k,v in q['criteria'].items()};rows.append({'state':r['state'],'question':q})
 (HERE/'merge-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 model,tok,_=load_model(str(ROOT/'releases/jet-v6.1'),'f446b82727be57da348bb46eccf211414294ab3e',adapter=str(ROOT/'adapters/jet-focused-20260925/lr3e-6/best'))
else:
 sys.path.insert(0,str(OUT));from jet import Jet
 wrapper=Jet(str(OUT));model,tok=wrapper.model,wrapper.tokenizer
 rows=list(map(json.loads,(HERE/'merge-cases.jsonl').open()))
refs=[];temps=json.loads((OUT/'calibration.json').read_text())
with torch.no_grad():
 for row in rows:
  q=Question.from_dict(row['question']);ids=encode(tok,row['state'],q,10**9);assert len(ids)<=16384
  z=label_logits(model,{'ids':ids,'labels':label_token_ids(tok,q)}).double();prob=(z/temps[q.type]).softmax(-1).cpu().tolist();refs.append({'type':q.type,'probabilities':prob})
path=FOCUS/'test.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((FOCUS/'final-report.json').read_text())['test_sha256']
holdout=evaluate(model,read_data(path,tok));(HERE/(mode+'-holdout.json')).write_text(json.dumps(holdout,indent=2)+'\n');print(mode,'holdout',json.dumps(holdout),flush=True)
if mode=='candidate':
 prior=json.loads((FOCUS/'final-report.json').read_text())['test']['selected_candidate']
 assert all(abs(holdout['families'][k]['metric']-v['metric'])<1e-12 for k,v in prior['families'].items()),'Candidate changed'
 (HERE/'candidate-probabilities.json').write_text(json.dumps(refs)+'\n')
else:
 prior=json.loads((HERE/'candidate-probabilities.json').read_text());assert len(prior)==len(refs)==157;records=[]
 for i,(a,b) in enumerate(zip(prior,refs)):
  ap=a['probabilities'];bp=b['probabilities'];assert len(ap)==len(bp)
  records.append({'case':i,'type':a['type'],'same_argmax':max(range(len(ap)),key=ap.__getitem__)==max(range(len(bp)),key=bp.__getitem__),'max_probability_difference':max(abs(x-y) for x,y in zip(ap,bp))})
 smoke=wrapper.decide('The item arrived broken.',{'category':{'type':'choice','instructions':'What happened?','criteria':{'damage':'The item was damaged.','billing':'A payment problem.'}},'severity':{'type':'score','instructions':'Rate severity.','criteria':['none','moderate','severe']},'damaged':{'type':'noul','instructions':'Is the item damaged?'}})
 assert set(smoke['answers'])=={'category','severity','damaged'}
 r=json.loads((FOCUS/'api-bank-longest.jsonl').read_text());long_smoke=wrapper.decide(r['state'],r['questions']);assert set(long_smoke['answers'])==set(r['questions'])
 try:Jet(str(OUT),max_tokens=16385)
 except ValueError:pass
 else:raise AssertionError('Context bound not enforced')
 base=json.loads((FOCUS/'final-report.json').read_text())['test']['released_v61']
 result={'cases':len(records),'argmax_flips':sum(not r['same_argmax'] for r in records),'max_probability_difference':max(r['max_probability_difference'] for r in records),'selection':'Previous fixed 144 merge cases plus 12 hash-selected focused holdout cases and longest API-Bank request. Not an independent benchmark.','calibration':'Inherited v6.1 temperatures, not refitted.','base_download_required':False,'wrapper_smoke':smoke,'context_probe':{'passed':True,'tokens':11495,'max_tokens':16384},'records':records,'merged_holdout':holdout,'released_holdout':base}
 result['merge_passed']=result['argmax_flips']/result['cases']<=.03 and result['max_probability_difference']<.075
 result['holdout_passed']=eligible(holdout,base) and holdout['objective']>base['objective']
 result['passed']=result['merge_passed'] and result['holdout_passed']
 (OUT/'merge-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','wrapper_smoke','merged_holdout','released_holdout']}),flush=True)
 assert result['passed'],'Release gates failed; keep published Jet v6.1'
