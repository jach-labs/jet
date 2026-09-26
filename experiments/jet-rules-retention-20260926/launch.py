"""Run serial GPU trials with frozen inputs and persist status outside Git."""
import datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'adapters'/HERE.name;LOGS=ROOT/'logs'/HERE.name
LOGS.mkdir(parents=True,exist_ok=True)
def status(**kw):
 p=HERE/'status.tmp';p.write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2)+'\n');p.replace(HERE/'status.json')
def run(name,args):
 with (LOGS/(name+'.log')).open('a') as log:
  process=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,TOKENIZERS_PARALLELISM='false'))
  status(state='running',phase=name,pid=process.pid,log=str(LOGS/(name+'.log')))
  code=process.wait()
 if code:raise RuntimeError(f'{name} failed: {code}')
try:
 status(state='verifying')
 audit=json.loads((HERE/'data-audit.json').read_text())
 for split in ['train','selection','test']:
  assert hashlib.sha256((HERE/(split+'.jsonl')).read_bytes()).hexdigest()==audit[split]['sha256']
 base=ROOT/'releases/jet-v6.2'
 for name,expected in json.loads((base/'release-manifest.json').read_text())['files'].items():
  if name.endswith('.safetensors'):
   with (base/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
 for path,expected in json.loads((HERE/'code-manifest.json').read_text()).items():
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
 run('teacher',[HERE/'teacher.py'])
 run('smoke',[HERE/'train.py','--lr','5e-6','--anchor','2','--out',OUT/'smoke','--smoke'])
 for name,strength in [('anchor2','2'),('anchor6','6')]:run(name,[HERE/'train.py','--lr','5e-6','--anchor',strength,'--out',OUT/name])
 run('final-test',[HERE/'final_test.py'])
 run('kev-comparison',[HERE/'compare_kev.py'])
 status(state='completed',report=str(HERE/'results.md'),published_model_unchanged=True)
except Exception as e:status(state='failed',error=str(e));traceback.print_exc();raise
