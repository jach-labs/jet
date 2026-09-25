"""Compare sampled Jet results with published full-suite benchmark references."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'leaderboard-snapshot.json').read_text())
ours=json.loads((HERE/'trained-benchmarks.json').read_text())
entrants=[d['jev']]+d['models']
selected=[d['jev']]+[next(m for m in d['models'] if m['engine']==e)
                      for e in ['decider-4b','kev-4b-raised','autojev-27b']]
panel={i for values in d['suite']['areas'].values() for i in values}
joined=[]
for b in ours['benchmarks']:
    for model in entrants:
        ref=model['results'].get(str(b['catalog_id']),{})
        match=b['score'] is not None and b['metric']==ref.get('metric') and ref.get('score') is not None
        joined.append({'dataset':b['dataset'],'catalog_id':b['catalog_id'],'metric':b['metric'],
            'jet_sample_requests':b['requests'],'jet_sample_score':b['score'],
            'reference_model':model['name'],'reference_engine':model['engine'],
            'reference_metric':ref.get('metric'),'reference_score':ref.get('score'),
            'reference_cases_or_requests':ref.get('requests',ref.get('cases')),
            'metric_matches':match,'in_current_panel':b['catalog_id'] in panel,
            'comparison_scope':'different denominators: local sample versus published full benchmark'})
(HERE/'leaderboard-comparison.json').write_text(json.dumps(joined,indent=2)+'\n')
with (HERE/'leaderboard-comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(joined[0]));writer.writeheader();writer.writerows(joined)

lines=['# Trained Jet 4B versus Decision Index entrants','',
       '**Jet numbers below are sampled diagnostics; other columns are published full-benchmark results. '
       'These are directional comparisons, not matched-case wins or an official Jet rank.**','',
       'Jet completed all 1900 requests across 15 datasets with zero errors or unsupported requests. '
       'Fourteen datasets have directly matching scalar metric names. The frozen model is step 3750, '
       'with calibration fitted only on calibration_v5. No benchmark result was used to tune it.', '',
       '| Dataset | Metric | Jet sample n | Jet 4B sample | Jev full | Decider 4B full | Kev 4B full | AutoJev-27B full |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for b in ours['benchmarks']:
    if b['score'] is None:continue
    values=[]
    for m in selected:
        r=m['results'].get(str(b['catalog_id']),{})
        values.append(f"{r['score']*100:.1f}" if r.get('score') is not None and r.get('metric')==b['metric'] else '—')
    star=' †' if b['catalog_id'] not in panel else ''
    lines.append(f"| {b['dataset']}{star} | {b['metric']} | {b['requests']} | {b['score']*100:.1f} | {' | '.join(values)} |")
lines += ['', 'Scores are displayed on a 0–100 scale. † MMLU and both ARC datasets are outside the current '
          '0.2 scored panel. Reference denominators differ and are recorded in leaderboard-comparison.csv. '
          'The mixed iSarcasm diagnostic is omitted from the comparison because the board uses English '
          'track-A sarcasm F1; combining its subtasks would be misleading.', '',
          '## What the diagnostic suggests', '',
          'Jet looks competitive with similarly sized entrants on mathematics and several entailment/classification '
          'tasks. Its strongest-looking comparisons include GSM8K, ContractNLI, NLI4CT and ANLI. '
          'The clearest areas to investigate are CRUXEval code behavior, WinoGrande commonsense, VAST stance '
          'and Habermas preferences. Small samples and different case mixes prevent claims of superiority.', '',
          '## Published overall Decision Index scores', '',
          '| Model | Published 0.2 index |', '|---|---:|', '| Jet 4B | Not measured |']
for m in selected:
    lines.append(f"| {m['name']} | {m['scores']['balanced_raw']:.2f} |")
lines += ['', 'No subset average is substituted for Jet’s missing overall score. The current index requires '
          '40 benchmarks across five equally weighted areas; this diagnostic does not cover that panel.', '',
          '## Remaining requirement for an official comparison', '',
          'Obtain the matching release-v2 corpus or rebuild recipe and scoring package. The Space references '
          '121057 requests and corpus SHA256 '
          '`b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`. The public reproduction '
          'repository currently documents the older edition and does not expose the referenced lab scripts. '
          'Once the exact suite is available, verify its hash and run its complete scoring protocol.', '',
          'The raw predictions, probability distributions, calibration files, code hashes, full benchmark summary '
          'and data/source audits are saved locally. No exact training-state matches were found in the diagnostic, '
          'but earlier source-contamination limitations remain; this is not a decontamination certification. '
          'No results or model weights were uploaded or submitted.', '',
          'The user redirected this task away from the untrained-base comparison. That diagnostic was stopped, '
          'and the planned local published-Jet run was not started. The comparison above uses the actual '
          'Decision Index entrants’ published results.', '',
          f"Leaderboard snapshot generated_utc: {d.get('generated_utc')}. Space revision: "
          '`953204c58869a05e911d5ec62b7588b1b97b03ba`.', '',
          '[Decision Index leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index) · '
          '[Published data](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json) · '
          '[Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json)', '',
          'See leaderboard-comparison.csv for comparisons against every published entrant, including reference '
          'sample counts and metric compatibility.']
(HERE/'results.md').write_text('\n'.join(lines)+'\n')
(HERE/'status.json').write_text(json.dumps({'state':'completed','trained_diagnostic_requests':1900,
    'leaderboard_entrants_compared':len(entrants),'matching_scalar_benchmarks':14,
    'base_benchmark':'stopped at user request','official_index':'not measured; matching release-v2 corpus unavailable locally'},indent=2)+'\n')
print('\n'.join(lines[:24]))
