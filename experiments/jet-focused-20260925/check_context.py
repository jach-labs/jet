"""Probe longest API-Bank input at 16K without truncation or reading its labels."""
import json,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'src'))
from decision_index_torch import TorchJetEngine
row=json.loads((HERE/'api-bank-longest.jsonl').read_text());start=time.monotonic()
engine=TorchJetEngine(model=str(ROOT/'releases/jet-v6.1'),revision='f446b82727be57da348bb46eccf211414294ab3e',calibration=str(ROOT/'releases/jet-v6.1/calibration.json'),max_tokens=16384)
result,raw=engine(row['state'],row['questions']);engine.synchronize()
report={'status':'ok','base_revision':engine.provenance['revision'],'max_tokens':16384,'prompt_tokens':{k:v['prompt_tokens'] for k,v in raw.items()},'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'seconds_including_load':time.monotonic()-start,'answer_keys':list(result['answers']),'scope':'Longest available API-Bank prompt, complete inputs and all options. Feasibility only; not a benchmark score.'}
(HERE/'api-bank-context-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
