import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from bench_decision_index import SampleSuite
from decision_index.scoring.report import load_results,benchmark_summary
for name in ['candidate','published']:
 p=ROOT/'artifacts/decision-index/runs'/HERE.name/name/'results.jsonl'
 if not p.exists():continue
 try: results=load_results(p)
 except Exception as e:print(e);continue
 summary=benchmark_summary(SampleSuite(HERE/'rows.jsonl'),results,name)
 print(name,summary['counts'])
 for b in summary['benchmarks']:
  print(json.dumps({k:b.get(k) for k in ['dataset','requests','answered','pending','errors','unsupported','metric','score','detail']}))
