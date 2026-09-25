"""Stage complete available benchmarks; never infer an overall from missing tasks."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
from decision_index.suite.io import Suite
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
method=json.loads((ROOT/'experiments/jet-4b-evaluation/methodology-v2.json').read_text())
panel={int(b['id']):b['name'] for a in method['index']['areas'] for b in a['panel']}
rows={}
for name in ['rebuild','v5-rebuild','esci-rebuild',*sys.argv[1:]]:
    path=ROOT/'artifacts/decision-index'/name/'artifacts/benchmark-suite/release-v1-rebuilt'
    for row in Suite(path).rows(apply_exclusions=True):
        e=row['_evaluation']
        if int(e['catalog_id']) not in panel:continue
        # The current headline for sarcasm uses English track A only.
        if int(e['catalog_id'])==40 and e['track']!='iSarcasmEval-A-En':continue
        rid=e['run_id']
        if rid in rows and rows[rid]['_evaluation']['payload_sha256']!=e['payload_sha256']:
            raise ValueError('Changed payload: '+rid)
        rows[rid]=row
out=HERE/('expanded.jsonl' if sys.argv[1:] else 'initial.jsonl')
with out.open('w') as f:
    for r in rows.values():f.write(json.dumps(r,separators=(',',':'))+'\n')
counts=Counter(int(r['_evaluation']['catalog_id']) for r in rows.values())
audit={'requests':len(rows),'rows_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'benchmarks':[{ 'id':k,'name':v,'requests':counts[k], 'state':'staged' if counts[k] else 'missing'} for k,v in panel.items()], 'official_overall':None,'note':'Full available v1 benchmarks; release-v2 ToolRet/BRIGHT selection and seven extension recipes remain unverified. No missing benchmark is assigned a fabricated score.'}
out.with_suffix('.manifest.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'rows':str(out),'requests':len(rows),'benchmarks':len(counts)}))
