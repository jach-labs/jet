"""Independently match every new V5 label back to its pinned native train row."""
import json,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import pyarrow.parquet as pq
from data.decision_training import ROOT,read,statekey

def main():
 specs=json.loads((ROOT/'work/jet-v5/sources-complete.json').read_text()); native=defaultdict(list);labels=None
 for s in specs:
  if s['role']!='train':continue
  if s['file'].endswith('.parquet'):
   t=pq.read_table(s['path']);rows=t.to_pylist()
   if 'clinc' in s['repo']:labels=json.loads(t.schema.metadata[b'huggingface'])['info']['features']['intent']['names']
  else:rows=[json.loads(x) for x in open(s['path'])]
  for i,r in enumerate(rows):
   key=f'{s["repo"]}@{s["revision"]}/{s["file"]}:{r.get("uid",r.get("ind",i))}'
   native[key].append(r)
 owners={}; counts=Counter(); origins=set()
 for split,file in [('train','train_v5'),('val','selection_v5'),('calibration','calibration_v5'),('test','test_v5_new')]:
  for row in read(ROOT/f'data/{file}.jsonl'):
   if not row['source'].startswith('v5:'):continue
   p=row['provenance']; assert p['origin'] not in origins; origins.add(p['origin'])
   answer=row['question']['criteria'][row['gold']]
   candidates=native[p['origin']];family=row['family']
   def matches(r):
    if family=='wide_intent':
     correct=labels[r['intent']]; correct='outside all supported intents' if correct=='oos' else correct.replace('_',' ')
     return row['state']==r['text'] and answer==correct and len(row['question']['criteria'])==151
    if family=='commonsense_completion':return row['state']==r['ctx'] and answer==r['endings'][int(r['label'])]
    if family=='adversarial_entailment':return row['state']==dict(premise=r['premise'],hypothesis=r['hypothesis']) and answer==['the premise entails the hypothesis','the premise does not determine the hypothesis','the premise contradicts the hypothesis'][r['label']]
    return row['state']['messages']==r['messages'] and answer==r['chosen_response']['content']
   assert any(matches(r) for r in candidates),(family,p['origin'])
   assert row['target'][list(row['question']['criteria']).index(row['gold'])]==1
   for key in [p['component'],statekey(row['state'])]+p['content_keys']:
    assert owners.setdefault(key,split)==split,(key,split)
   counts[split+':'+family]+=1
 report={'verified_native_labels':sum(counts.values()),'counts':dict(counts),'source_id_duplicates':0,'cross_split_related_groups':0}
 (ROOT/'docs/training/jet-v5/data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
