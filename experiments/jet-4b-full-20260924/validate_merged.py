import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
sys.path.insert(0,str(Path('releases/jet-v6').resolve()))
from jet import Jet
from format import Question,label_token_ids
from inference import encode
from runtime import label_logits
model=Jet();data=[json.loads(l) for l in Path('data/test_v5_new.jsonl').open()];old=json.loads(Path('experiments/jet-4b-evaluation/trained/test_v5_new-logits.json').read_text())
assert len(data)==len(old)==600
data += [json.loads(l) for l in Path('data/calibration_v5.jsonl').open()]
old += json.loads(Path('experiments/jet-4b-evaluation/trained/calibration_v5-logits.json').read_text())
counts=Counter();records=[]
with torch.no_grad():
 for i,(row,ref) in enumerate(zip(data,old)):
  kind=row['question']['type']
  if counts[kind]>=12:continue
  counts[kind]+=1;q=Question.from_dict(row['question'])
  assert kind==ref['type'] and row['source']==ref['source']
  ids=encode(model.tokenizer,row['state'],q,10**9)
  z=label_logits(model.model,{'ids':ids,'labels':label_token_ids(model.tokenizer,q)}).double()
  a=torch.tensor(ref['logits'],device='cuda',dtype=torch.float64)
  p=(z/model.temperatures[kind]).softmax(-1);original=(a/model.temperatures[kind]).softmax(-1)
  records.append({'test_index':i,'type':kind,'same_argmax':int(z.argmax())==int(a.argmax()),'max_probability_difference':float((p-original).abs().max())})
 # Exercise the public wrapper on all types without using private data in release files.
 result=model.decide('The item arrived broken.',{'category':{'type':'choice','instructions':'What happened?','criteria':{'damage':'The item was damaged.','billing':'A payment problem.'}},'severity':{'type':'score','instructions':'Rate the severity.','criteria':['none','moderate','severe']},'damaged':{'type':'noul','instructions':'Is the item damaged?'}})
 assert set(result['answers'])=={'category','severity','damaged'}
report={'cases':len(records),'selection':'First 12 cases per type, scanning local test then calibration; no outcome selection','argmax_flips':sum(not r['same_argmax'] for r in records),'max_probability_difference':max(r['max_probability_difference'] for r in records),'records':records,'wrapper_smoke':result,'base_download_required':False}
Path('releases/jet-v6/merge-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['records','wrapper_smoke']}))
assert report['argmax_flips']==0 and report['max_probability_difference']<.05
