"""Require completed runs, verify preserved files, and archive reproducibility metadata."""
import hashlib
import csv
import json
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/training/jet-next'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for name in ['warm','v4']:
        for prefix in ['diagnostic','esci','isarcasm-en','cruxeval-full']:
            r=json.loads((OUT/f'{prefix}-{name}.json').read_text())
            assert r['complete_for_supplied_rows'] and r['pending_requests']==0
            assert r['summary']['counts']=={'ok':r['planned_requests']},r['summary']['counts']
    runs={}
    for name,directory in [('warm','jet-v3-warm-20260924'),('v4','jet-v4-transfer-20260924')]:
        p=ROOT/'adapters'/directory;c=json.loads((p/'training_config.json').read_text())
        metrics=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
        assert metrics[-1]['step']==c['total_steps']
        for filename,expected in c['data_sha256'].items():assert sha(ROOT/filename)==expected
        assert sha(ROOT/c['init_adapter']/'adapters.safetensors')==c['init_adapter_sha256']
        best=min(metrics,key=lambda m:m['nll'])
        runs[name]=dict(directory=str(p.relative_to(ROOT)),config=c,best=best,
            weights_sha256=sha(p/'adapters.safetensors'),last_weights_sha256=sha(p/'last/adapters.safetensors'),
            calibration=json.loads((p/'calibration.json').read_text()))
        for f in ['training_config.json','metrics.jsonl','calibration.json']:
            shutil.copy2(p/f,OUT/f'{name}-{f.replace("_", "-")}')
    for name in ['protected','published']:
        subprocess.run(['sha256sum','-c',str(ROOT/f'work/jet-v3/{name}.sha256')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    current=subprocess.check_output(['git','diff','--','README.md','src/data/distill.py'],cwd=ROOT)
    assert current==(ROOT/'work/jet-v3/preexisting.patch').read_bytes(),'Preexisting user changes differ'
    code=['src/train.py','src/evaluate.py','src/model.py','src/inference.py','src/format.py','src/decision_index_engine.py','src/bench_decision_index.py','src/data/decision_training.py','train_cuda.sh','scripts/bench_index_cuda.sh','scripts/evaluate_next_code.sh']
    code += ['scripts/'+n for n in ['build_v4_transfer.py','audit_v4_transfer.py','prepare_v4_focus.py','build_v4_tools.py','audit_v4_tools.py','run_next_cuda.sh','evaluate_next_extra.sh','summarize_next.py','finalize_next.py']]
    for relative in code:
        dest=OUT/'code'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    hashes={p:sha(ROOT/p) for p in code+['uv.lock','pyproject.toml']}
    corpus={p.name:sha(p) for p in (ROOT/'data').glob('*v4*.jsonl')}
    samples=list(csv.DictReader((ROOT/'logs/jet-next/gpu.csv').open(),skipinitialspace=True))
    peak=max(int(r['memory.used [MiB]'].split()[0]) for r in samples)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    manifest=dict(status='completed',base='mlx-community/Qwen3-0.6B-bf16',base_revision='42096995f6402fde107068cf530136fe64b604f8',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        runs=runs,hardware=hardware,monitored_device_peak_mib=peak,monitor_scope='V4 training and subsequent evaluations, including desktop GPU use',code_sha256=hashes,data_sha256=corpus,protected_original_files='unchanged',published_model='unchanged',preexisting_user_changes='unchanged',
        warm_source_snapshot_sha256=sha(OUT/'train-warm-snapshot.py'),
        limitations=['No official full-suite or leaderboard score','Native metrics only; installed panel proposal differs from deployed panel','Partial diagnostic differences are descriptive, not significance tests'])
    (OUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    destination=Path('/home/jach/Documents/Codex/2026-09-23/cl/outputs');destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/'run-manifest.json',destination/'jet-next-run-manifest.json')
    shutil.copy2(OUT/'comparison.json',destination/'jet-next-results.json')
    print('Completed runs verified; published model, original files and prior changes preserved.')
if __name__=='__main__':main()
