"""Run full available tasks on frozen Jet; no base-model run or uploads."""
import json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1')
RUN=ROOT/'artifacts/decision-index/runs/jet-4b-full-index'

def status(**kw):(HERE/'status.json').write_text(json.dumps(kw,indent=2)+'\n')
def run(rows):
    status(state='running',rows=str(rows),official_overall=None)
    args=[sys.executable,'-m','decision_index','run','--engine','decision_index_torch:TorchJetEngine','--rows',str(rows),'--out',str(RUN),'--compact']
    options={'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','adapter':str(ROOT/'adapters/jet-4b-full-20260924/best'),'calibration':str(ROOT/'experiments/jet-4b-evaluation/trained/calibration.json'),'max_tokens':8192}
    for k,v in options.items():args+=['--option',f'{k}={v}']
    with (ROOT/'logs/jet-4b-full-index'/('run-'+rows.stem+'.log')).open('w') as log:
        subprocess.run(args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    sys.path.insert(0,str(ROOT/'src'))
    from bench_decision_index import SampleSuite
    from decision_index.scoring.report import load_results,benchmark_summary
    summary=benchmark_summary(SampleSuite(rows),load_results(RUN/'results.jsonl'),'Jet 4B')
    (HERE/(rows.stem+'-scores.json')).write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    try:
        run(HERE/'initial.jsonl')
        # The rebuild runs independently and writes a completion marker.
        import time
        while True:
            p=HERE/'rebuild-status.json'
            state=json.loads(p.read_text()) if p.exists() else {}
            if state.get('state')=='complete':break
            if state.get('state')=='failed':raise RuntimeError('Rebuild failed; initial results retained')
            time.sleep(30)
        subprocess.run([sys.executable,str(HERE/'prepare.py'),'full-index-rebuild'],cwd=ROOT,env=ENV,check=True)
        run(HERE/'expanded.jsonl')
        status(state='available_benchmarks_completed',official_overall=None,note='Check staged panel coverage and release-v2 input availability before computing the current index.')
    except Exception as e:
        status(state='failed',error=str(e),official_overall=None);traceback.print_exc();raise
