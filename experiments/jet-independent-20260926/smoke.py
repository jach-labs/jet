"""Exercise the merged release's own loader and public typed API."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'releases/jet-rules-candidate-20260926'
sys.path.insert(0,str(OUT))
from jet import Jet
j=Jet(str(OUT));result=j.decide('The item arrived broken.',{'category':{'type':'choice','instructions':'What happened?','criteria':{'damage':'The item was damaged.','billing':'A payment problem.'}},'severity':{'type':'score','instructions':'Rate severity.','criteria':['none','moderate','severe']},'damaged':{'type':'noul','instructions':'Is the item damaged?'}})
assert set(result['answers'])=={'category','severity','damaged'}
row=json.loads((ROOT/'experiments/jet-focused-20260925/api-bank-longest.jsonl').read_text());long=j.decide(row['state'],row['questions']);assert set(long['answers'])==set(row['questions'])
try:Jet(str(OUT),max_tokens=16385)
except ValueError:pass
else:raise AssertionError('Context bound not enforced')
(HERE/'runtime-smoke.json').write_text(json.dumps({'passed':True,'typed_api':result,'long_context_passed':True,'context_tokens':11495,'max_tokens':16384,'context_limit_enforced':True},indent=2)+'\n')
print('Merged runtime smoke passed',flush=True)
