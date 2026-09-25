import datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def status(state,**kw):
 p=HERE/'status.tmp';p.write_text(json.dumps({'state':state,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2)+'\n');p.replace(HERE/'status.json')
try:
 previous=json.loads((HERE/'status.json').read_text())
 command=previous['command']
 if '--resume' not in command:command.append('--resume')
 base=ROOT/'releases/jet-v6'
 for name,expected in json.loads((base/'release-manifest.json').read_text())['files'].items():
  if name.endswith('.safetensors'):
   with (base/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
 log=ROOT/'logs'/HERE.name/'train.log'
 with log.open('a') as stream:
  process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
  status('train',pid=process.pid,command=command,log=str(log),resumed_from_step=2250)
  code=process.wait()
 if code:raise RuntimeError(f'Training exited {code}')
 status('completed',published_release_unchanged=True)
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();raise
