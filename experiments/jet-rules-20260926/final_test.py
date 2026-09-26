"""Freeze a validation-selected winner, then evaluate its untouched focus holdout once."""
import gc,hashlib,json,sys
from pathlib import Path
import torch
from train import HERE,ROOT,load_model,read_data,evaluate,eligible
OUT=ROOT/'adapters'/HERE.name
REV='fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a'
def main():
 trials=['lr5e-6','lr15e-6'];selection={}
 for t in trials:
  assert (OUT/t/'completed.json').exists()
  selection[t]=json.loads((OUT/t/'best-step.json').read_text())
 winner=max(trials,key=lambda t:selection[t]['objective']);decision={'winner':winner,'selection':selection,'selected_before_test':True,'base_revision':REV}
 (HERE/'selection-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 report={**decision,'test_sha256':hashlib.sha256((HERE/'test.jsonl').read_bytes()).hexdigest(),'test':{},'published_model_unchanged':True}
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
 for name,adapter in [('released_v62',None),('selected_candidate',OUT/winner/'best')]:
  model,tok,_=load_model(str(ROOT/'releases/jet-v6.2'),REV,adapter=str(adapter) if adapter else None)
  report['test'][name]=evaluate(model,read_data(HERE/'test.jsonl',tok));print(name,json.dumps(report['test'][name]),flush=True)
  del model,tok;gc.collect();torch.cuda.empty_cache()
 base=report['test']['released_v62'];candidate=report['test']['selected_candidate']
 report['passes_final_guard']=selection[winner]['step']>0 and eligible(candidate,base) and candidate['objective']>base['objective']
 report['next_action']='Merge and validate this frozen candidate before any release.' if report['passes_final_guard'] else 'Keep Jet v6.2; no candidate passed the final improvement guard.'
 report['limitations']=json.loads((HERE/'data-audit.json').read_text())['limitations']
 (HERE/'final-report.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Focused Jet v6.2 continuation','',f'Validation-selected trial: {winner}, step {selection[winner]["step"]}.','', '| Family | Released v6.2 | Candidate |','|---|---:|---:|']
 for family in base['families']:
  lines.append(f'| {family} | {100*base["families"][family]["metric"]:.2f} | {100*candidate["families"][family]["metric"]:.2f} |')
 lines+=['',report['next_action'],'','Focus holdouts are new; retention holdout is reused. No official Decision Index score is produced.','',*report['limitations']]
 (HERE/'results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
