"""Unlock fresh test once both short trials have selected their checkpoints."""
import json,sys,hashlib,gc
from pathlib import Path
import torch
from train import HERE,ROOT,load_model,read_data,evaluate
OUT=ROOT/'adapters/jet-repair-20260925'
def main():
 trials=['lr5e-6','lr1e-5'];selection={}
 for t in trials:
  assert (OUT/t/'completed.json').exists()
  selection[t]=json.loads((OUT/t/'best-step.json').read_text())
 winner=max(trials,key=lambda t:selection[t]['objective'])
 report={'winner_selected_before_test':winner,'selection':selection,'test_sha256':hashlib.sha256((HERE/'test.jsonl').read_bytes()).hexdigest(),'fresh_test':{},'published_model_unchanged':True}
 (HERE/'selection-decision.json').write_text(json.dumps({'winner':winner,'selection':selection},indent=2)+'\n')
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
 models={'published':None,'previous_candidate':ROOT/'adapters/jet-targeted-20260924/best',**{t:OUT/t/'best' for t in trials}}
 for name,adapter in models.items():
  model,tok,_=load_model(str(ROOT/'releases/jet-v6'),'e5b8f610ddb92ffaba596ae452bed32a9fef49ca',adapter=str(adapter) if adapter else None)
  data=read_data(HERE/'test.jsonl',tok);report['fresh_test'][name]=evaluate(model,data)
  print(name,json.dumps(report['fresh_test'][name]),flush=True)
  del model,tok,data;gc.collect();torch.cuda.empty_cache()
 report['limitations']=json.loads((HERE/'data-audit.json').read_text())['limitations']
 (HERE/'final-report.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Jet repair trials','',f'Selected by validation before opening the fresh test: **{winner}**.','', '| Task / metric | Published | Previous candidate | 5e-6 | 1e-5 |','|---|---:|---:|---:|---:|']
 for task,v in report['fresh_test']['published']['families'].items():
  values=[report['fresh_test'][name]['families'][task]['metric']*100 for name in models]
  lines.append('| '+task+' / '+v['metric_name']+' | '+' | '.join(f'{x:.2f}' for x in values)+' |')
 lines+=['','Fresh source holdouts; these are not the Decision Index benchmark scores. Published Jet is unchanged.','',*report['limitations']]
 (HERE/'results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
