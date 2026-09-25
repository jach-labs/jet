"""Audit native-evaluation overlap in retained examples and create immutable R2 data.

Catalog names and isolated short schema fields are not treated as utterances.
V5 source content identities, complete text states and text fragments >=32
normalized characters are compared; inherited exposure is reported explicitly.
"""
import json
from pathlib import Path
from collections import Counter
from build_v5 import records,content_keys,sha
from data.decision_training import ROOT,read,write,atoms,digest,normalized

def keys(row):
 if row['source'].startswith('v5:'):return set(row['provenance']['content_keys'])
 result={'content:'+a for a in atoms(row['state'])}
 if isinstance(row['state'],str):result.add('content:'+digest(normalized(row['state'])))
 return result

def main():
 blocked=set()
 for s in json.loads((ROOT/'work/jet-v5/sources-complete.json').read_text()):
  if s['role']!='train':
   for r in records(s):blocked.update(content_keys(s['repo'],r))
 report={'rule':__doc__,'files':{},'inherited_exposure':{}}
 for name in ['train_v2','train_v3','train_v4_tools_focus']:
  hits=[r for r in read(ROOT/f'data/{name}.jsonl') if keys(r)&blocked]
  report['inherited_exposure'][name]={'count':len(hits),'by_source':dict(Counter(r['source'] for r in hits)),'state_hashes':[digest(r['state']) for r in hits]}
 for name in ['train_v5','selection_v5','calibration_v5','test_v5_new']:
  src=ROOT/f'data/{name}.jsonl';rows=read(src)
  kept=[r for r in rows if not keys(r)&blocked];hits=[r for r in rows if keys(r)&blocked]
  if name=='train_v5':
   dst=ROOT/'data/train_v5_r2.jsonl';assert not dst.exists();write(dst,kept)
   report['output']={'path':str(dst),'sha256':sha(dst),'rows':len(kept)}
  else:assert not hits,(name,len(hits))
  report['files'][name]={'sha256':sha(src),'kept':len(kept),'removed':len(hits),'removed_sources':dict(Counter(r['source'] for r in hits)),'state_hashes':[digest(r['state']) for r in hits]}
 smoke=read(ROOT/'work/jet-v5/smoke.jsonl');assert not any(keys(r)&blocked for r in smoke)
 report['caveat']='R2 removes new-stage overlap but cannot erase inherited V4 training exposure; do not claim a fully decontaminated checkpoint or an official score.'
 (ROOT/'docs/training/jet-v5/retention-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
