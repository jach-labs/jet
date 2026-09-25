"""Validate and train a new candidate; never restart the stopped benchmark."""
import datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LOGS=ROOT/'logs'/HERE.name;OUT=ROOT/'adapters'/HERE.name
BASE=ROOT/'releases/jet-v6'
def status(state,**kw):
 x={'state':state,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw};p=HERE/'status.tmp';p.write_text(json.dumps(x,indent=2)+'\n');p.replace(HERE/'status.json')
try:
 status('verifying_base')
 manifest=json.loads((BASE/'release-manifest.json').read_text())
 for name,expected in manifest['files'].items():
  if name.endswith('.safetensors'):
   with (BASE/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
 common=[sys.executable,'-u',str(HERE/'code/train.py'),'--base',str(BASE),'--revision','e5b8f610ddb92ffaba596ae452bed32a9fef49ca','--train',str(HERE/'train.jsonl'),'--val',str(HERE/'selection.jsonl'),'--epochs','1','--lr','0.00002','--rank','16','--accumulate','4','--eval-every','250','--max-prompt','3072','--seed','24092417']
 for stage in ['smoke','train']:
  out=Path(str(OUT)+'-smoke') if stage=='smoke' else OUT
  command=common+['--out',str(out)]+(['--smoke'] if stage=='smoke' else [])
  with (LOGS/(stage+'.log')).open('w') as log:
   process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   status(stage,pid=process.pid,log=str(LOGS/(stage+'.log')),command=command)
   code=process.wait()
  if code:raise RuntimeError(f'{stage} failed with exit code {code}')
 status('completed',candidate=str(OUT/'best'),published_release_unchanged=True)
except Exception as e:
 status('failed',error=str(e));traceback.print_exc();raise
