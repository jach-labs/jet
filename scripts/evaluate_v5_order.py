"""Choose single versus two-order readout using selection data only."""
import argparse,json,random,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from decision_index_ensemble import TwoOrderJetEngine

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--adapter',required=True);args=ap.parse_args()
 rows=[json.loads(x) for x in Path('data/selection_v5.jsonl').read_text().splitlines()]
 groups=defaultdict(list)
 for r in rows:
  if r['question']['type']=='choice':groups[r.get('family',r['source'])].append(r)
 for k in sorted(groups):random.Random(240925).shuffle(groups[k])
 selected=[]
 while groups and len(selected)<600:
  for k in list(groups):
   selected.append(groups[k].pop())
   if not groups[k]:del groups[k]
   if len(selected)==600:break
 engine=TwoOrderJetEngine(model='mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8',adapter=args.adapter,max_tokens=8192,max_cached_tokens=8192)
 results=[];started=time.monotonic()
 for i,r in enumerate(selected):
  response,raw=engine(r['state'],{'q':r['question']})
  keys=list(r['question']['criteria']);target=np.array(r['target'])
  scores={}
  for name,answer in [('single',raw['single_order_answers']['q']),('two_order',response['answers']['q'])]:
   p=np.array([answer['probabilities'][k] for k in keys])
   scores[name]={'nll':float(-(target*np.log(np.clip(p,1e-12,1))).sum()),'correct':int(p.argmax()==target.argmax())}
  results.append({'family':r.get('family',r['source']),'scores':scores})
  if (i+1)%25==0:print(i+1,len(selected),flush=True)
 summary={}
 for name in ['single','two_order']:
  families=defaultdict(list)
  for r in results:families[r['family']].append(r['scores'][name])
  summary[name]={'nll':float(np.mean([r['scores'][name]['nll'] for r in results])),
   'accuracy':float(np.mean([r['scores'][name]['correct'] for r in results])),
   'macro_family_accuracy':float(np.mean([np.mean([r['correct'] for r in rs]) for rs in families.values()]))}
 choose=summary['two_order']['nll']<summary['single']['nll'] and summary['two_order']['macro_family_accuracy']>=summary['single']['macro_family_accuracy']
 report={'n':len(results),'selection_sha256':hashlib.sha256(Path('data/selection_v5.jsonl').read_bytes()).hexdigest(),'selection_rule':'lower NLL and no decrease in macro family accuracy on selection data only','selected':'two_order' if choose else 'single','summary':summary,'seconds':time.monotonic()-started,'engine_provenance':engine.provenance,'rows':results}
 Path('docs/training/jet-v5/inference-selection.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['rows','engine_provenance']},indent=2))
if __name__=='__main__':main()
