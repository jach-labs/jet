"""Cache frozen v6.2 predictions on training inputs, never selection/test inputs."""
import hashlib,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'src'))
from qwen35_training import load_model,examples,label_logits
assert not (HERE/'teacher-logits.json').exists()
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
model,tok,_=load_model(str(ROOT/'releases/jet-v6.2'));data=examples(HERE/'train.jsonl',tok,3072);rows=[]
with torch.no_grad():
 for i,ex in enumerate(data):
  key=hashlib.sha256(json.dumps([ex['ids'],ex['labels']]).encode()).hexdigest()
  rows.append({'key':key,'logits':label_logits(model,ex).cpu().tolist()})
  if (i+1)%500==0:print(f'Teacher cached {i+1}/{len(data)}',flush=True)
cache={'base_revision':'fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a','train_sha256':hashlib.sha256((HERE/'train.jsonl').read_bytes()).hexdigest(),'rows':rows}
(HERE/'teacher-logits.json').write_text(json.dumps(cache)+'\n')
(HERE/'teacher-receipt.json').write_text(json.dumps({k:v for k,v in cache.items() if k!='rows'}|{'rows':len(rows),'cache_sha256':hashlib.sha256((HERE/'teacher-logits.json').read_bytes()).hexdigest()},indent=2)+'\n')
