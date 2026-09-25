"""Summarize completed, fixed-test comparisons without a reconstructed index."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'docs/training/jet-v3';OUT=ROOT/'docs/training/jet-next'
MODELS=['baseline','v3','warm','v4'];LABELS=['Released Jet','V3','Warm start','V4 transfer']
def load(path):return json.loads(path.read_text())
def pct(x):return '—' if x is None else f'{100*x:.1f}'
def main():
    data={}
    for dataset in ['test','score_eval','test_v3_new','test_v4_new','test_v4_tools']:
        data[dataset]={m:load(OUT/f'{m}-{dataset}.json') for m in MODELS}
    diagnostics={m:load((OLD if m in ['baseline','v3'] else OUT)/f'diagnostic-{m}.json') for m in MODELS}
    esci={m:load(OUT/f'esci-{m}.json') for m in MODELS}
    sarcasm={m:load(OUT/f'isarcasm-en-{m}.json') for m in MODELS}
    code={m:load(OUT/f'cruxeval-full-{m}.json') for m in MODELS}
    for group in [diagnostics,esci,sarcasm,code]:
        assert len({r['rows_sha256'] for r in group.values()})==1
        assert all(r['complete_for_supplied_rows'] and r['pending_requests']==0 for r in group.values())
    results=dict(local_tests=data,diagnostics=diagnostics,esci=esci,isarcasm_english=sarcasm,cruxeval_full=code)
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2))
    lines=['# Jet Qwen3-0.6B follow-up results','',
        'Two more local checkpoints were trained: a lower-learning-rate warm start from released Jet, then targeted transfer training. No weights were published and no leaderboard entry was submitted.',
        '', 'V4 substantially improves contract entailment, product relevance, stance, and English sarcasm over released Jet on the evaluations below. Confirmed regressions remain in CRUXEval and original-test accuracy. The warm checkpoint preserves original-test accuracy more closely. Neither candidate is an across-the-board replacement.',
        '', '**These results do not establish a leaderboard win.** The official frozen suite remains unavailable. Diagnostics are partial public reconstructions; the installed kit also contains an older panel definition. Native benchmark metrics below are comparable across these four models on identical rows, but are not an overall index.',
        '', '| Held-out accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |', '|---|---:|---:|---:|---:|']
    names=dict(test='Original test (3,683)',score_eval='Ordinal test (2,400)',test_v3_new='V3 new families (1,364)',test_v4_new='Harder transfer tasks (720)',test_v4_tools='SGD domain-held-out routing (240)')
    for ds,reports in data.items():lines.append('| '+names[ds]+' | '+' | '.join(pct(reports[m]['overall']['acc']) for m in MODELS)+' |')
    lines += ['', '| Partial benchmark — native metric (%) | Released Jet | V3 | Warm start | V4 transfer |','|---|---:|---:|---:|---:|']
    for group in [diagnostics,esci]:
        tables={m:{r['catalog_id']:r for r in report['summary']['benchmarks']} for m,report in group.items()}
        for bid,base in tables['baseline'].items():
            lines.append('| '+base['dataset']+' — '+str(base.get('metric') or 'unavailable')+' | '+' | '.join(pct(tables[m][bid].get('score')) for m in MODELS)+' |')
    english={m:next(t for t in r['panel_metrics_on_sample']['40']['tracks'] if t['track']=='iSarcasmEval-A-En')['raw'] for m,r in sarcasm.items()}
    lines.append('| iSarcasmEval English A, all 1,400 cases — positive F1 | '+' | '.join(pct(english[m]) for m in MODELS)+' |')
    lines.append('| CRUXEval, all '+str(code['baseline']['planned_requests'])+' available eligible cases — accuracy | '+' | '.join(pct(code[m]['summary']['benchmarks'][0]['score']) for m in MODELS)+' |')
    lines += ['', 'iSarcasmEval has no aggregate native score in the pinned summary report. A missing value is not zero. The separate English-A row uses the native positive-class F1 scorer on all 1,400 available cases (the official headline track), without combining other sarcasm tracks. The 500-request eight-benchmark sample and separate 500-request ESCI sample are small; differences are descriptive, not statistical significance claims.',
        '', '| V4 task family — accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |','|---|---:|---:|---:|---:|']
    for ds in ['test_v4_new','test_v4_tools']:
        for family in sorted(data[ds]['baseline']['by_source']):
            lines.append('| '+family.removeprefix('v4:')+' | '+' | '.join(pct(data[ds][m]['by_source'][family]['acc']) for m in MODELS)+' |')
    lines += ['', '| Original family — accuracy (%) | Released Jet | V3 | Warm start | V4 transfer |','|---|---:|---:|---:|---:|']
    for family in sorted(data['test']['baseline']['by_source']):
        lines.append('| '+family+' | '+' | '.join(pct(data['test'][m]['by_source'][family]['acc']) for m in MODELS)+' |')
    lines += ['', '| Calibration metric | Released Jet | V3 | Warm start | V4 transfer |','|---|---:|---:|---:|---:|']
    for ds in ['test','test_v3_new','test_v4_new','test_v4_tools']:
        for metric in ['nll','ece']:
            lines.append('| '+ds+' '+metric.upper()+' | '+' | '.join(f"{data[ds][m]['overall'][metric]:.4f}" for m in MODELS)+' |')
    for metric in ['spearman','mae']:
        lines.append('| Ordinal test '+metric+' | '+' | '.join(f"{data['score_eval'][m]['by_type']['score'][metric]:.4f}" for m in MODELS)+' |')
    lines += ['', 'Both candidates were selected by held-out validation NLL and calibrated on separate data before these test runs. Original three test sets use the same 4,096-token state limit for all models; new V4 tests use 2,048 for all models. Benchmark adapters preserve complete prompts up to 8,192 tokens.', '', '## Training and provenance', '']
    for name,directory in [('Warm start','jet-v3-warm-20260924'),('V4 transfer','jet-v4-transfer-20260924')]:
        p=ROOT/'adapters'/directory;c=load(p/'training_config.json');metrics=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
        best=min(metrics,key=lambda r:r['nll'])
        lines.append(f"- {name}: `{p.relative_to(ROOT)}`; {c['train_examples']:,} rows, {c['total_steps']:,} steps, LR {c['lr']}, best validation step {best['step']}, NLL {best['nll']:.4f}, accuracy {best['acc']:.4f}. Weight SHA256 `{hashlib.sha256((p/'adapters.safetensors').read_bytes()).hexdigest()}`.")
    lines += ['', 'The published model remains unchanged. V4 retains 18,000 earlier training examples and adds 3,000 examples each for full-contract entailment, stance, executable multi-step code, and native service-intent routing. Sources are pinned; original training labels were verified; related documents/articles/domains stay together. No hosted model APIs were used.', '', 'Full experimental protocol and reproduction details follow.', '', (ROOT/'docs/training/jet-next.md').read_text()]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    destination=Path('/home/jach/Documents/Codex/2026-09-23/cl/outputs/jet-next-report.md');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text('\n'.join(lines)+'\n')
    print(destination)
if __name__=='__main__':main()
