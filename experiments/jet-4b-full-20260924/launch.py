"""Detached sequential capacity check and full training, with durable status."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOGS = ROOT / 'logs' / HERE.name
BASE = ROOT / 'adapters' / HERE.name


def status(state, **extra):
    value = {'state': state, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'supervisor_pid': os.getpid(), **extra}
    temp = HERE / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(HERE / 'status.json')


common = [sys.executable, '-u', str(HERE / 'code/train.py'),
          '--base', 'Qwen/Qwen3.5-4B', '--revision', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
          '--train', 'data/train_v5_r2.jsonl', '--val', 'data/selection_v5.jsonl',
          '--epochs', '1', '--lr', '0.0001', '--rank', '16', '--accumulate', '4',
          '--eval-every', '250', '--max-prompt', '3072', '--seed', '240924']

try:
    for stage in ['smoke', 'train']:
        out = Path(str(BASE) + '-smoke') if stage == 'smoke' else BASE
        command = common + ['--out', str(out)] + (['--smoke'] if stage == 'smoke' else [])
        log_path = LOGS / (stage + '.log')
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                                       stdin=subprocess.DEVNULL)
            status(stage, training_pid=process.pid, log=str(log_path), adapter_dir=str(out), command=command)
            result = process.wait()
        if result:
            status('failed', failed_stage=stage, exit_code=result, log=str(log_path))
            sys.exit(result)
    status('completed', adapter_dir=str(BASE), log=str(LOGS / 'train.log'))
except Exception as error:
    status('failed', error=repr(error))
    raise
