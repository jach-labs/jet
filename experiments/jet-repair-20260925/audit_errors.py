import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[2]
rows={r['_evaluation']['run_id']:r for r in map(json.loads,(root/'experiments/jet-targeted-eval-20260925/rows.jsonl').open()) if r['_evaluation']['dataset']=='iSarcasmEval'}
report={}
for name in ['published','candidate']:
 c=collections.Counter()
 for r in map(json.loads,(root/f'artifacts/decision-index/runs/jet-targeted-eval-20260925/{name}/results.jsonl').open()):
  if r['run_id'] not in rows:continue
  gold=rows[r['run_id']]['expected']['sarcastic'];pred=r['response']['answers']['sarcastic']['choice'];c[('T' if gold==pred else 'F')+('P' if pred=='yes' else 'N')]+=1
 report[name]=dict(c)
 report[name].update(precision=c['TP']/(c['TP']+c['FP']),recall=c['TP']/(c['TP']+c['FN']))
Path(__file__).with_name('sarcasm-error-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
