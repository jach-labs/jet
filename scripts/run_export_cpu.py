"""Run the production ONNX exporter on CPU with a resident-memory watchdog."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
command = [sys.executable, '-m', 'optimum.commands.optimum_cli', 'export', 'onnx', '--model', str(root/'models/jet-v4'), '--task', 'text-generation-with-past', '--device', 'cpu', '--dtype', 'fp32', '--no-post-process', '--batch_size', '1', '--sequence_length', '16', str(root/'artifacts/release-v4/export')]
proc = subprocess.Popen(command, env=env, start_new_session=True)
peak = 0
while proc.poll() is None:
    status = Path(f'/proc/{proc.pid}/status')
    try:
        rss = next(int(line.split()[1]) for line in status.read_text().splitlines() if line.startswith('VmRSS:'))
        peak = max(peak, rss)
        if rss > 8*1024*1024:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit('Export exceeded 8 GiB resident-memory limit')
    except (FileNotFoundError, StopIteration):
        pass
    time.sleep(0.5)
print(f'Peak exporter resident memory: {peak/1024:.0f} MiB', flush=True)
sys.exit(proc.returncode)
