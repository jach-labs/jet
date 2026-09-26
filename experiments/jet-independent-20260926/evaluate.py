"""Score fixed fresh and reused panels with one frozen candidate; never search checkpoints."""
import hashlib,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PARENT=ROOT/'experiments/jet-rules-retention-20260926';OUT=ROOT/'releases/jet-rules-candidate-20260926'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(PARENT))
from train import read_data,evaluate
from qwen35_training import load_model,label_logits
MODE=sys.argv[1];assert MODE in ['baseline','candidate','merged']
ADAPTER=ROOT/'adapters/jet-rules-retention-20260926/anchor2/best'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER/'adapter_model.safetensors')=='3ad3cc233db094723e875f978bee59de402f8901e0761630e4a9dad14e59f89f'
assert sha(HERE/'fresh.jsonl')==json.loads((HERE/'data-audit.json').read_text())['sha256']
assert sha(PARENT/'test.jsonl')==json.loads((PARENT/'data-audit.json').read_text())['test']['sha256']
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
model,tok,_=load_model(str(OUT if MODE=='merged' else ROOT/'releases/jet-v6.2'),adapter=str(ADAPTER) if MODE=='candidate' else None)
report={'mode':MODE,'panels':{},'adapter_sha256':sha(ADAPTER/'adapter_model.safetensors')};predictions={}
for name,path in [('fresh',HERE/'fresh.jsonl'),('reused',PARENT/'test.jsonl')]:
 data=read_data(path,tok);report['panels'][name]=evaluate(model,data);predictions[name]=[]
 if name=='fresh':
  with torch.no_grad():
   for ex in data:
    z=label_logits(model,ex);predictions[name].append(z.double().softmax(-1).cpu().tolist())
 print(MODE,name,json.dumps(report['panels'][name]),flush=True)
(HERE/(MODE+'-metrics.json')).write_text(json.dumps(report,indent=2)+'\n')
(HERE/(MODE+'-probabilities.json')).write_text(json.dumps(predictions)+'\n')
