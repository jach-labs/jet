"""Compare published Kev scores with complete local results; label unmatched samples."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from bench_decision_index import SampleSuite
from decision_index.scoring.report import benchmark_summary,load_results

def main():
 protocol=json.loads((HERE/'protocol.json').read_text())
 results=ROOT/'artifacts/decision-index/runs'/HERE.name/'results.jsonl'
 summary=benchmark_summary(SampleSuite(HERE/'rows.jsonl'),load_results(results),'Jet candidate step 2000')
 (HERE/'jet-scores.json').write_text(json.dumps(summary,indent=2)+'\n')
 byid={int(b['catalog_id']):b for b in summary['benchmarks']};comparison=[]
 for spec in protocol['datasets']:
  b=byid.get(spec['id'],{});score=b.get('score');metric=b.get('metric');reference=spec['reference_score']
  if spec['id']==40:
   score=b.get('detail',{}).get('positive_f1_by_field',{}).get('sarcastic') if b.get('detail') else None;metric='Sarcasm F1 · track A, English'
  complete=b.get('pending',spec['requests'])==0
  metric_match=metric==spec['reference_metric']
  count_match=spec['requests']==spec['reference_requests']
  compatible=complete and metric_match and count_match and spec['scope']=='full available reconstruction' and b.get('answered')==spec['requests'] and not b.get('errors',0) and not b.get('unsupported',0)
  row={**spec,'jet_score':score,'local_metric':metric,'complete':complete,'answered':b.get('answered',0),'unsupported':b.get('unsupported',0),'errors':b.get('errors',0),'pending':b.get('pending',spec['requests']),'count_match':count_match,'metric_match':metric_match,'indicative_comparison':compatible,'delta_pp':100*(score-reference) if compatible and score is not None and reference is not None else None}
  comparison.append(row)
 report={'counts':summary['counts'],'comparison':comparison,'official_overall':None,'limitations':protocol['limitations'],'reference_url':protocol['reference_url']}
 (HERE/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Jet versus Kev 8B — broader development comparison','',f'Reference: [archived Decision Index 0.1]({protocol["reference_url"]}). Jet is the earlier step-2,000 candidate on the published merged backbone.','', 'Matching metrics/counts are indicative comparisons; frozen case identity is unverified. Missing/partial results are not wins or zero scores.','', '| Benchmark | Jet | Kev reference | Difference (pp) | Local / reference cases | Status |','|---|---:|---:|---:|---:|---|']
 for r in comparison:
  score=f'{100*r["jet_score"]:.2f}' if r['complete'] and r['jet_score'] is not None else 'pending' if not r['complete'] else 'unscored'
  ref=f'{100*r["reference_score"]:.2f}' if r['reference_score'] is not None else '—';delta=f'{r["delta_pp"]:+.2f}' if r['delta_pp'] is not None else '—'
  status='complete; metric/count match' if r['indicative_comparison'] else 'pending' if not r['complete'] else 'sample diagnostic' if r['scope'].startswith('sampled') else 'metric or denominator differs'
  if r['errors'] or r['unsupported']:status+=f'; {r["errors"]} errors, {r["unsupported"]} unsupported'
  lines.append(f'| {r["dataset"]} | {score} | {ref} | {delta} | {r["requests"]} / {r["reference_requests"]} | {status} |')
 eligible=[r for r in comparison if r['delta_pp'] is not None and not r['errors'] and not r['unsupported']]
 if eligible:
  lines+=['','## Largest measured deficits','']
  deficits=sorted((r for r in eligible if r['delta_pp']<-.005),key=lambda r:r['delta_pp'])[:5]
  lines += [f'- {r["dataset"]}: {r["delta_pp"]:.2f} percentage points below the published reference.' for r in deficits] or ['No deficits among completed, comparable entries.']
 lines+=['','## Scope','',*['- '+s for s in protocol['limitations']]]
 (HERE/'results.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'counts':summary['counts'],'completed_benchmarks':sum(r['complete'] for r in comparison),'total_benchmarks':len(comparison)}))
if __name__=='__main__':main()
