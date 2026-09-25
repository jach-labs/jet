"""Verify V5 completion, original-file preservation, and selected artifact hashes."""
import csv,hashlib,json,math,shutil,subprocess,platform
from importlib.metadata import version
from pathlib import Path
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/training/jet-v5'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 adapter=ROOT/'adapters/jet-v5-panel02-r2-20260924'
 config=json.loads((adapter/'training_config.json').read_text())
 metrics=[json.loads(x) for x in (adapter/'metrics.jsonl').read_text().splitlines()]
 assert metrics[-1]['step']==config['total_steps']
 best=min(metrics,key=lambda r:r['nll'])
 with safe_open(adapter/'optimizer.safetensors',framework='numpy') as f:step=int(f.get_tensor('step'))
 assert step==best['step'],(step,best)
 temperatures=json.loads((adapter/'calibration.json').read_text())
 assert all(math.isfinite(t) and t>0 for t in temperatures.values())
 data=json.loads((D/'retention-audit.json').read_text())
 assert sha(ROOT/'data/train_v5_r2.jsonl')==data['output']['sha256']
 for name,meta in data['files'].items():assert sha(ROOT/f'data/{name}.jsonl')==meta['sha256']
 original_provenance=json.loads((ROOT/'data/train_v5.provenance.json').read_text())
 for path,digest in original_provenance['protected'].items():assert sha(Path(path))==digest,path
 for name in ['protected','published']:
  subprocess.run(['sha256sum','-c',str(ROOT/f'work/jet-v3/{name}.sha256')],cwd=ROOT,check=True)
 counts={}
 for suite in ['diagnostic','v5-diagnostic-1400']:
  path=ROOT/f'artifacts/decision-index/runs/v5-compatibility-{suite}/results.jsonl'
  rows=[json.loads(s) for s in path.read_text().splitlines()]
  statuses=[r['status'] for r in rows];assert all(s=='ok' for s in statuses),statuses
  counts[suite]=len(rows)
 code=D/'final-code';code.mkdir(exist_ok=True)
 names=['src/train.py','src/model.py','src/evaluate.py','src/format.py','src/inference.py','src/decision_index_engine.py','src/decision_index_ensemble.py']
 names += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*v5*.py')]
 names += ['scripts/run_v5_cuda.sh','scripts/run_v5_extra.sh','tests/test_decision_ensemble.py']
 code_hashes={}
 for name in names:
  p=ROOT/name;q=code/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);code_hashes[name]=sha(p)
 gpu=list(csv.DictReader((ROOT/'logs/jet-v5/gpu.csv').open()))
 peak=max(int(r[' memory.used [MiB]'].strip().split()[0]) for r in gpu)
 environment={'platform':platform.platform(),'packages':{p:version(p) for p in ['mlx','mlx-lm','numpy','transformers','huggingface-hub','decision-index']},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()}
 report={'environment':environment,'selected_step':step,'total_steps':config['total_steps'],'adapter_sha256':sha(adapter/'adapters.safetensors'),'calibration':temperatures,'compatibility_ok':counts,'whole_session_gpu_peak_mib':peak,'training_config':config,'code_sha256':code_hashes,'dataset':data,'published_model_unchanged':True,'old_validation_test_unchanged':True,'scope':'Partial diagnostics; inherited overlap caveat applies. No official Decision Index 0.2 score.'}
 (D/'completion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 out=Path('/home/jach/Documents/Codex/2026-09-23/cl/outputs')
 shutil.copy2(D/'completion-audit.json',out/'jet-v5-run-manifest.json')
 print(json.dumps({k:v for k,v in report.items() if k not in ['training_config','code_sha256','dataset']},indent=2))
if __name__=='__main__':main()
