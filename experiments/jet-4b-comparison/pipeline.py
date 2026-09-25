"""Wait for pinned local weights, then run the pilots sequentially on one GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
models = json.loads((HERE / 'models.json').read_text())
pending = list(range(len(models)))
deadline = time.monotonic() + 3600
while pending:
    if time.monotonic() > deadline:
        raise TimeoutError('Pinned model weights did not become available within one hour')
    ready = None
    for i in pending:
        model = models[i]
        snapshot = Path.home() / '.cache/huggingface/hub' / ('models--' + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']
        index = snapshot / 'model.safetensors.index.json'
        if index.exists() and all((snapshot / f).exists() for f in set(json.loads(index.read_text())['weight_map'].values())):
            ready = i
            break
    if ready is None:
        time.sleep(5)
        continue
    for mode in ['smoke', 'run']:
        print(f'START {models[ready]["repo"]} {mode}', flush=True)
        with (ROOT / 'logs/jet-4b-comparison' / f'model-{ready}-{mode}.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'run.py'), mode, '--model', str(ready)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'FINISHED {models[ready]["repo"]} {mode}', flush=True)
    pending.remove(ready)
subprocess.run([sys.executable, str(HERE / 'report.py')], cwd=ROOT, check=True)
