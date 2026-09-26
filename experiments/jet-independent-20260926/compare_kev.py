"""One final development-suite pass for the validation-selected candidate only."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REF=ROOT/'experiments/jet-kev-family-20260926'
sys.path.insert(0,str(ROOT/'src'))
from qwen35_training import load_model,label_logits
from format import label_token_ids
from inference import encode
spec=importlib.util.spec_from_file_location('frozen_family_conversion',REF/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);convert=module.convert
selection=json.loads((ROOT/'experiments/jet-rules-retention-20260926/selection-decision.json').read_text());winner=selection['winner'];adapter=ROOT/'adapters/jet-rules-retention-20260926'/winner/'best'
assert (HERE/'merged-metrics.json').exists()
protocol=json.loads((REF/'protocol.json').read_text());assert hashlib.sha256((REF/'suite.jsonl').read_bytes()).hexdigest()==protocol['suite_sha256']
rows=[r for r in map(json.loads,(REF/'suite.jsonl').open()) if r['_meta']['variant']=='clean'];assert len(rows)==656
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
model,tok,_=load_model(str(ROOT/'releases/jet-rules-candidate-20260926'));temps=json.loads((ROOT/'releases/jet-v6.2/calibration.json').read_text());by=defaultdict(list)
with torch.no_grad():
 for row in rows:
  state,q,target,task=convert(row);ids=encode(tok,state,q,10**9);assert len(ids)<=16384
  z=label_logits(model,{'ids':ids,'labels':label_token_ids(tok,q)}).double();p=(z/temps[q.type]).softmax(-1).cpu().numpy();by[task].append((int(p.argmax()==np.argmax(target)),float(((p-np.array(target))**2).sum())))
all_rows=[r for rs in by.values() for r in rs];baseline=json.loads((REF/'jet-results.json').read_text());reference=json.loads((REF/'kev-reference.json').read_text())['models']['kev-4b']
scores={k:float(np.mean([r[0] for r in rs])) for k,rs in by.items()};accuracy=float(np.mean([r[0] for r in all_rows]));targets=['composition_held_and_or','composition_held_or_not','composition_held_conditional','contrastive_deadline']
report={'trial':winner,'step':selection['selection'][winner]['step'],'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),'suite_sha256':protocol['suite_sha256'],'n':len(all_rows),'accuracy':accuracy,'brier':float(np.mean([r[1] for r in all_rows])),'tasks':scores,'baseline_accuracy':baseline['transfer_acc'],'baseline_tasks':baseline['tasks'],'kev4b_accuracy':reference['transfer_acc'],'kev4b_tasks':reference['tasks'],'matches_or_beats_kev4b_on_targets':all(scores[k]>=reference['tasks'][k] for k in targets),'matches_or_beats_kev4b_overall':accuracy>=reference['transfer_acc'],'limitations':['Benchmark-informed development: targeted families are now trained, so not an out-of-domain claim.','Kev figures are frozen published reference scores, not newly run models.','Candidate is the locally merged BF16 checkpoint; it has not been published.']}
(HERE/'kev-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
