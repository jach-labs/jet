import datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];LOG=ROOT/'logs'/HERE.name;LOG.mkdir(exist_ok=True)
def status(**kw):
 p=HERE/'status.tmp';p.write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2)+'\n');p.replace(HERE/'status.json')
try:
 for path,expected in json.loads((HERE/'code-manifest.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
 base=ROOT/'releases/jet-v6.2'
 for name,expected in json.loads((base/'release-manifest.json').read_text())['files'].items():
  if name.endswith('.safetensors'):
   with (base/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
 for phase,args in [('baseline',['evaluate.py','baseline']),('candidate',['evaluate.py','candidate']),('merge',['merge.py']),('merged',['evaluate.py','merged']),('runtime-smoke',['smoke.py']),('kev-comparison',['compare_kev.py']),('report',['report.py'])]:
  with (LOG/(phase+'.log')).open('a') as f:
   p=subprocess.Popen([sys.executable,'-u',str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,TOKENIZERS_PARALLELISM='false'))
   status(state='running',phase=phase,pid=p.pid);code=p.wait()
  if code:raise RuntimeError(f'{phase} exited {code}')
 status(state='completed',report=str(HERE/'results.md'),published_model_unchanged=True)
except Exception as e:status(state='failed',error=str(e));traceback.print_exc();raise
