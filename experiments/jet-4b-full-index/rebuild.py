"""Rebuild available pinned public tasks independently, retaining failure evidence."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from decision_index.suite.build.layout import Layout,NAMES
from decision_index.suite.build.rebuild import BUILDERS,main as freeze
from decision_index.suite.build.acquire import acquire
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORK=ROOT/'artifacts/decision-index/full-index-rebuild'
layout=Layout(WORK)

def save(data):
    tmp=HERE/'rebuild-status.tmp'
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(HERE/'rebuild-status.json')

def failure(kind, error, tb):
    if len(sys.argv)==1:save({'state':'failed','error':str(error)})
    sys.__excepthook__(kind,error,tb)
sys.excepthook=failure

if len(sys.argv)>1:
    ids=list(map(int,sys.argv[1].split(',')))
    acquire(layout,ids)
    # Upstream acquisition and normalizers disagree on the raw/ directory.
    if (WORK/'raw').exists():shutil.copytree(WORK/'raw',layout.raw,dirs_exist_ok=True)
    BUILDERS[ids[0]](layout)
else:
    for name in ['rebuild','v5-rebuild','esci-rebuild']:
        source=ROOT/'artifacts/decision-index'/name
        for rel in ['artifacts/benchmark-suite/raw','artifacts/benchmark-suite/normalized','data/sources','raw']:
            if not (source/rel).exists():continue
            dest=WORK/rel;dest.mkdir(parents=True,exist_ok=True)
            subprocess.run(['cp','-a','--update=none','--reflink=auto',str(source/rel)+'/.',str(dest)],check=True)
    method=json.loads((ROOT/'experiments/jet-4b-evaluation/methodology-v2.json').read_text())
    panel={b['id'] for a in method['index']['areas'] for b in a['panel']}
    groups={}
    for n,fn in BUILDERS.items():groups.setdefault(fn,[]).append(n)
    state={'state':'building','groups':{},'missing_recipes':sorted(panel-set(BUILDERS))}
    for fn,ids in groups.items():
        if not panel.intersection(ids):continue
        paths=[layout.normalized/(name+'.jsonl') for n in ids for name in NAMES[n]]
        if all(p.exists() and p.stat().st_size for p in paths):
            state['groups'][fn.__name__]={'state':'cached','ids':ids};save(state);continue
        state['active']=fn.__name__;save(state)
        try:
            with (ROOT/'logs/jet-4b-full-index'/('build-'+fn.__name__+'.log')).open('w') as log:
                r=subprocess.run([sys.executable,__file__,','.join(map(str,ids))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            ok=r.returncode==0 and all(p.exists() and p.stat().st_size for p in paths)
            state['groups'][fn.__name__]={'state':'built' if ok else 'failed','ids':ids,'exit_code':r.returncode}
            if not ok:
                for p in paths:p.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            state['groups'][fn.__name__]={'state':'timeout','ids':ids}
            for p in paths:p.unlink(missing_ok=True)
        save(state)
    result=freeze(WORK,skip_download=True,skip_normalize=True)
    shutil.copy2(ROOT/'artifacts/decision-index/rebuild/artifacts/benchmark-suite/release-v1-rebuilt/excluded-questions.json',Path(result['out'])/'excluded-questions.json')
    state.update(state='complete',active=None,result=result);save(state)
