"""Apply preregistered gates without selecting another checkpoint."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def passes(candidate,baseline):
 return candidate['objective']>baseline['objective'] and all(candidate['families'][k]['metric']>=v['metric']-.02 for k,v in baseline['families'].items())

def main():
 reports={mode:json.loads((HERE/(mode+'-metrics.json')).read_text())['panels'] for mode in ['baseline','candidate','merged']}
 a=json.loads((HERE/'candidate-probabilities.json').read_text())['fresh'];b=json.loads((HERE/'merged-probabilities.json').read_text())['fresh'];assert len(a)==len(b)
 drift={'n':len(a),'argmax_flips':sum(max(range(len(x)),key=x.__getitem__)!=max(range(len(y)),key=y.__getitem__) for x,y in zip(a,b)),'max_probability_difference':max(abs(p-q) for x,y in zip(a,b) for p,q in zip(x,y))}
 audit=json.loads((HERE/'data-audit.json').read_text());sarcasm_n=audit['class_counts']['sarcasm'].get('sarcastic',0)
 gates={'candidate_fresh':passes(reports['candidate']['fresh'],reports['baseline']['fresh']),'merged_fresh':passes(reports['merged']['fresh'],reports['baseline']['fresh']),'candidate_reused':passes(reports['candidate']['reused'],reports['baseline']['reused']),'merged_reused':passes(reports['merged']['reused'],reports['baseline']['reused']),'merge_fidelity':drift['argmax_flips']/drift['n']<=.03 and drift['max_probability_difference']<.075,'runtime_smoke':json.loads((HERE/'runtime-smoke.json').read_text())['passed'],'fresh_sarcasm_sample_sufficient':sarcasm_n>=20}
 report={'gates':gates,'release_ready':all(gates.values()),'merge':drift,'panels':reports,'fresh_sarcasm_positives':sarcasm_n,'limitations':audit['limitations'],'published_model_unchanged':True}
 (HERE/'final-report.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Independent candidate validation','', '| Fresh metric | Released v6.2 | Adapter | Merged BF16 |','|---|---:|---:|---:|']
 for k,v in reports['baseline']['fresh']['families'].items():lines.append('| '+k+' | '+' | '.join(f'{reports[m]["fresh"]["families"][k]["metric"]*100:.2f}' for m in ['baseline','candidate','merged'])+' |')
 lines+=['','## Gates','',*[f'- {k}: {v}' for k,v in gates.items()],'',f'Merge: {drift["argmax_flips"]}/{drift["n"]} argmax flips; maximum raw probability difference {drift["max_probability_difference"]:.6f}.','',*audit['limitations'],'','Published Jet remains v6.2. No official Decision Index score is produced.']
 kev=json.loads((HERE/'kev-comparison.json').read_text());lines+=['',f'Merged Kev development accuracy: {kev["accuracy"]*100:.2f}%; released Jet {kev["baseline_accuracy"]*100:.2f}%; published Kev-4B {kev["kev4b_accuracy"]*100:.2f}%.']
 (HERE/'results.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'gates':gates,'release_ready':report['release_ready'],'merge':drift}),flush=True)
if __name__=='__main__':main()
