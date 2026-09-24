"""Summarize completed V5 evaluation without inventing a leaderboard score."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/training/jet-v5'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 adapter=ROOT/'adapters/jet-v5-panel02-r2-20260924'
 config=read(adapter/'training_config.json');metrics=[json.loads(x) for x in (adapter/'metrics.jsonl').read_text().splitlines()]
 assert metrics[-1]['step']==config['total_steps'],'Training incomplete'
 order=read(D/'inference-selection.json')
 held={name:{ds:read(D/f'{name}-{ds}.json') for ds in ['test','test_v5_new','test_v4_new','test_v4_tools']} for name in ['v4','v5']}
 bench={}
 for name in ['v4','v5']:
  bench[name]={}
  for suite in ['diagnostic','expanded']:
   result=read(D/f'{name}-{suite}.json');assert result['complete_for_supplied_rows'] and result['pending_requests']==0
   for b in result['summary']['benchmarks']:bench[name][b['dataset']]=b
 if order['selected']=='two_order':
  bench['v5_two_order']={}
  for suite in ['original','expanded']:
   result=read(D/f'v5-two-order-{suite}.json');assert result['complete_for_supplied_rows']
   for b in result['summary']['benchmarks']:bench['v5_two_order'][b['dataset']]=b
 best=min(metrics,key=lambda r:r['nll'])
 lines=['# Jet V5 results — Qwen3-0.6B','',
 'V5 was trained locally on 15,997 examples: 7,997 retained V4 rows plus 2,000 each for CLINC150, ANLI, HellaSwag, and When2Call response preference. Native training partitions only; new selection, calibration and test groups are separate.','',
 '**These are partial diagnostics, not an official Decision Index 0.2 score or a leaderboard win.** The current board scores 40 benchmarks; the public reproduction kit still specifies the older panel. No submission was made.','',
 f'Selected training step: {best["step"]}/{config["total_steps"]}; selection NLL {best["nll"]:.4f}, accuracy {100*best["acc"]:.2f}%. Rank 16, LR 1e-5, one epoch, 3,072 batch tokens, 2,048 training-state tokens.','',
 '| Held-out accuracy (%) | V4 | V5 | Change (points) |','|---|---:|---:|---:|']
 for ds in ['test','test_v5_new','test_v4_new','test_v4_tools']:
  a=held['v4'][ds]['overall']['acc'];b=held['v5'][ds]['overall']['acc'];lines.append(f'| {ds} | {100*a:.2f} | {100*b:.2f} | {100*(b-a):+.2f} |')
 lines+=['','| New held-out family | V4 | V5 |','|---|---:|---:|']
 for family,a in held['v4']['test_v5_new']['by_source'].items():
  b=held['v5']['test_v5_new']['by_source'][family];lines.append(f'| {family} | {100*a["acc"]:.2f} | {100*b["acc"]:.2f} |')
 lines+=['','## Native benchmark metrics on fixed partial samples','',
 '| Benchmark | Metric | Requests | V4 | V5 | V5 two-order |','|---|---|---:|---:|---:|---:|']
 def fmt(x):return '—' if x is None else f'{100*x:.2f}'
 for name,a in bench['v4'].items():
  b=bench['v5'][name];c=bench.get('v5_two_order',{}).get(name,{})
  lines.append(f'| {name} | {a["metric"]} | {a["requests"]} | {fmt(a["score"])} | {fmt(b["score"])} | {fmt(c.get("score"))} |')
 lines+=['','All reported values are native metrics multiplied by 100. A missing aggregate remains missing. Samples are small and results are descriptive, not statistical significance claims. MMLU and ARC are diagnostic-only and are not in the current 0.2 index panel.','',
 f'Option-order policy chosen on {order["n"]} selection-only choice rows: **{order["selected"]}**. The two-order engine averages original/reversed probabilities using exact option keys; it keeps complete inputs and approximately doubles forward-pass work. Selection required lower NLL without lower macro family accuracy.','',
 'The first V5 attempt was stopped because three retained utterances overlapped native evaluation content. R2 removes them from continuation data. The V4 initialization still has inherited cross-dataset content overlap; this checkpoint is not certified decontaminated. See retention-audit.json.', '',
 'Single-order metrics remain separately reported. No test or benchmark score selected training steps, temperatures, or the option-order policy. Existing validation/test files and published weights were preserved.','',
 'The data protocol, source revisions and limitations are recorded in protocol.md and data/train_v5.provenance.json. Native annotations and group/source-ID separation were independently audited.']
 (D/'summary.md').write_text('\n'.join(lines)+'\n')
 result={'selected_checkpoint':best,'training_config':config,'adapter_sha256':sha(adapter/'adapters.safetensors'),'calibration':read(adapter/'calibration.json'),'heldout':held,'benchmarks':bench,'inference_selection':order,'data_provenance_sha256':sha(ROOT/'data/train_v5.provenance.json'),'retention_audit':read(D/'retention-audit.json'),'r2_training_sha256':sha(ROOT/'data/train_v5_r2.jsonl')}
 (D/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 out=Path('/home/jach/Documents/Codex/2026-09-23/cl/outputs');out.mkdir(parents=True,exist_ok=True)
 shutil.copy2(D/'summary.md',out/'jet-v5-report.md');shutil.copy2(D/'results.json',out/'jet-v5-results.json')
 print('\n'.join(lines))
if __name__=='__main__':main()
