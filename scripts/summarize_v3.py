"""Write the completed local experiment scorecard; never compute a leaderboard index."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
reports=root/'docs/training/jet-v3'
load=lambda p:json.loads(p.read_text())
run=root/'adapters/jet-v3-decision-20260923'
metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
config=load(run/'training_config.json');assert metrics[-1]['step']==config['total_steps']
best=min(metrics,key=lambda x:x['nll'])
provenance=load(reports/'data-provenance.json')
lines=['# Jet v3 completed experiment','',
'Qwen3-0.6B, Arch Linux, RTX 4080 SUPER (16 GB). Local training only; released Jet unchanged.','',
f"Completed **{config['total_steps']:,} steps / {config['epochs']:g} epochs**, with {config['train_examples']:,} training rows. Best checkpoint: step {best['step']:,}, selection NLL {best['nll']:.4f}, accuracy {best['acc']:.1%}.",
'',f"New adapter: `{run}`.",f"Calibration: `{json.dumps(load(run/'calibration.json'))}`.",'',
'## Independent evaluation','',
'| Set | Rows | Released Jet accuracy | v3 accuracy | Change | Baseline → v3 NLL | Baseline → v3 ECE |',
'|---|---:|---:|---:|---:|---:|---:|']
comparison={}
for dataset in ['test','score_eval','test_v3_new']:
    b=load(reports/f'baseline-{dataset}.json');v=load(reports/f'v3-{dataset}.json')
    comparison[dataset]={'baseline':b,'v3':v};a,c=b['overall'],v['overall']
    lines.append(f"| {dataset} | {a['n']} | {a['acc']:.1%} | {c['acc']:.1%} | {(c['acc']-a['acc'])*100:+.1f} pp | {a['nll']:.3f} → {c['nll']:.3f} | {a['ece']:.3f} → {c['ece']:.3f} |")
b,v=comparison['test_v3_new']['baseline'],comparison['test_v3_new']['v3']
public=[k for k in b['by_source'] if k not in {'v3:arithmetic','v3:boolean_rules','v3:code_behavior'}]
n=sum(b['by_source'][k]['n'] for k in public)
bacc=sum(b['by_source'][k]['n']*b['by_source'][k]['acc'] for k in public)/n
vacc=sum(v['by_source'][k]['n']*v['by_source'][k]['acc'] for k in public)/n
lines+=['',f'Excluding the local Boolean, arithmetic, and code generators, the {n}-row public-source subset improves from **{bacc:.1%} to {vacc:.1%}**. The Glaive part is itself publicly released synthetic data.']
for dataset in ['test','score_eval','test_v3_new']:
    b,v=comparison[dataset]['baseline'],comparison[dataset]['v3']
    lines+=['',f'### {dataset}: source breakdown','',
            '| Source | Rows | Released Jet | v3 | Change | NLL before → after |',
            '|---|---:|---:|---:|---:|---:|']
    for source in sorted(b['by_source']):
        a,c=b['by_source'][source],v['by_source'][source]
        lines.append(f"| {source} | {a['n']} | {a['acc']:.1%} | {c['acc']:.1%} | {(c['acc']-a['acc'])*100:+.1f} pp | {a['nll']:.3f} → {c['nll']:.3f} |")
    lines+=['',f"Batched latency: released fused model {b['ms_per_question']:.1f} ms/question; v3 unfused adapter {v['ms_per_question']:.1f} ms/question. This is not a fusion-controlled latency comparison."]
    if dataset=='score_eval':
        a,c=b['by_type']['score'],v['by_type']['score']
        lines+=['',f"Ordinal within-one accuracy: {a['within1']:.1%} → {c['within1']:.1%}; Spearman: {a['spearman']:.3f} → {c['spearman']:.3f}; normalized MAE: {a['mae']:.3f} → {c['mae']:.3f}."]
lines+=['','## Partial Decision Index diagnostic','',
'**Not a leaderboard score.** Same 500 complete-group requests from the eight-benchmark public rebuild, complete native prompts up to 8,192 tokens. Frozen official suite remains unavailable.','',
'| Benchmark | Metric | Released Jet | v3 | Requests |','|---|---|---:|---:|---:|']
bd,vd=[load(reports/f'diagnostic-{name}.json') for name in ['baseline','v3']]
for d in [bd,vd]:assert d['complete_for_supplied_rows'] and d['pending_requests']==0
bs={r['dataset']:r for r in bd['summary']['benchmarks']}
for r in vd['summary']['benchmarks']:
    if not r['requests']:continue
    b=bs[r['dataset']]
    fmt=lambda value: 'not provided' if value is None else f'{value:.4f}'
    lines.append(f"| {r['dataset']} | {r['metric']} | {fmt(b['score'])} | {fmt(r['score'])} | {r['requests']} |")
lines+=['', 'The pinned official summary provides no native iSarcasmEval subtask scores (null). Its secondary field accuracy rises from 57.3% to 66.1%, but category positive-F1 falls from 0.0317 to 0.0136; neither is substituted for a native headline score.', '', 'The synthetic code held-out gain does not transfer to CRUXEval here (40.0% → 38.2%). Likewise, the stance hold-out gain does not transfer to VAST (macro-F1 0.366 → 0.323). Keep these generator/domain limitations explicit.', '', 'Recommendation: retain released Jet as the default. Keep v3 as a data-expansion experiment; the gains on new tasks come with regressions on existing tests and mixed out-of-domain transfer. Do not tune against these final test results or promote this as a leaderboard result.']
lines+=['','## Audit and limits','',
'- New training examples come from pinned public training partitions and deterministic executable generators. Original IDs, hashes, grouping, and configuration are saved.',
'- All original data/evaluation files and published model files were preserved. 46 original training rows sharing held-out states were removed only from train_v3 (21 validation overlaps, 25 test overlaps). These are state-level overlaps, not necessarily identical questions.',
'- New families use separate checkpoint-selection, calibration, and final-test components. Existing validation was divided by state into selection and calibration without editing it.',
'- New training is balanced at 2,000 rows per family except stance (1,982). Tool tests are small after strict connected grouping; treat their changes as noisy diagnostics.',
'- Public Glaive annotations and distractor selection can contain semantic ambiguity. Product data uses one ESCI training shard. Synthetic tests assess these generator distributions, not general reasoning mastery.',
'- Training uses the existing 1,024-token state policy; overlong new rows were excluded. The Decision Index adapter preserves complete evaluation prompts.',
'- This single run changes the data mixture and its held-out selection/calibration mix. It does not establish a replicated causal gain or a full-suite score.',
'', '## Exact configuration','', '```json',json.dumps(config,indent=2),'```','',
f"Logs: `{root/'logs/jet-v3'}`. Data provenance and score JSON: `{reports}`.",
f"Implementation and source descriptions: `{root/'docs/training/jet-v3.md'}`."]
(reports/'summary.md').write_text('\n'.join(lines)+'\n')
(reports/'comparison.json').write_text(json.dumps(comparison,indent=2))
print(reports/'summary.md')
