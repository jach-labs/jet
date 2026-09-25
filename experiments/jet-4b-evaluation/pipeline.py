"""Run remaining calibration and three sequential, resumable diagnostic passes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1')
ROWS=ROOT/'artifacts/decision-index/jet-4b-evaluation/diagnostic.jsonl'
RUNS=ROOT/'artifacts/decision-index/runs/jet-4b-evaluation'


def command(stage,args):
    (HERE/'status.json').write_text(json.dumps({'stage':stage,'state':'running'})+'\n')
    with (ROOT/'logs/jet-4b-evaluation'/f'{stage}.log').open('w') as log:
        subprocess.run([sys.executable,*args],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    command('calibrate-base',[str(HERE/'calibrate.py'),'base'])
    for variant in ['trained','base','published-jet']:
        if variant=='published-jet':
            options={'model':'michaljach/jet','revision':'ee7d33ecae2b8bee30891657dd57c92d34e0ce53'}
        else:
            options={'model':'Qwen/Qwen3.5-4B','revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                     'calibration':str(HERE/variant/'calibration.json')}
            if variant=='trained':options['adapter']=str(ROOT/'adapters/jet-4b-full-20260924/best')
        args=['-m','decision_index','run','--engine','decision_index_torch:TorchJetEngine',
              '--rows',str(ROWS),'--out',str(RUNS/variant),'--option','max_tokens=8192']
        for k,v in options.items():args += ['--option',f'{k}={v}']
        command('benchmark-'+variant,args)
        from bench_decision_index import SampleSuite
        from decision_index.scoring.report import load_results,benchmark_summary
        results=load_results(RUNS/variant/'results.jsonl')
        summary=benchmark_summary(SampleSuite(ROWS),results,variant)
        assert len(results)==1900, f'Incomplete run: {variant}'
        if summary['counts'].get('error',0):raise RuntimeError(f'Benchmark errors: {summary["counts"]}')
        (HERE/(variant+'-benchmarks.json')).write_text(json.dumps(summary,indent=2)+'\n')
    command('report',[str(HERE/'report.py')])
    (HERE/'status.json').write_text(json.dumps({'state':'completed','requests_per_model':1900})+'\n')

if __name__=='__main__':
    try:main()
    except Exception as error:
        (HERE/'status.json').write_text(json.dumps({'state':'failed','error':repr(error)})+'\n')
        raise
