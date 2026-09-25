"""Fit only on calibration_v5; evaluate frozen models on separate test_v5_new."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from qwen35_training import load_model, examples, label_logits

HERE = Path(__file__).resolve().parent
ADAPTER = ROOT/'adapters/jet-4b-full-20260924/best'


def nll(z, target, temperature):
    z = np.asarray(z, dtype=float)/temperature
    z -= z.max()
    return float(-(np.asarray(target)*(z-np.log(np.exp(z).sum()))).sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('variant',choices=['trained','base']);args=parser.parse_args()
    out=HERE/args.variant;out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88)
    model,tok,_=load_model('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                         adapter=str(ADAPTER) if args.variant=='trained' else None)
    collected={};started=time.monotonic()
    for split in ['calibration_v5','test_v5_new']:
        rows=examples(ROOT/'data'/f'{split}.jsonl',tok,8192);records=[]
        with torch.no_grad():
            for i,ex in enumerate(rows):
                z=label_logits(model,ex).cpu().tolist()
                records.append({'type':ex['type'],'source':ex['source'],'target':ex['target'],'logits':z})
                if (i+1)%100==0:print(args.variant,split,i+1,len(rows),flush=True)
        collected[split]=records
        (out/(split+'-logits.json')).write_text(json.dumps(records)+'\n')
    temperatures={};fit={}
    grid=np.exp(np.linspace(np.log(.25),np.log(8.),121))
    for kind in ['choice','score','noul']:
        group=[r for r in collected['calibration_v5'] if r['type']==kind]
        scores=[np.mean([nll(r['logits'],r['target'],t) for r in group]) for t in grid]
        t=float(grid[np.argmin(scores)]) if group else 1.
        temperatures[kind]=t
        fit[kind]={'n':len(group),'temperature':t,'raw_nll':float(np.mean([nll(r['logits'],r['target'],1.) for r in group])) if group else None,'calibrated_nll':float(min(scores)) if group else None}
    (out/'calibration.json').write_text(json.dumps(temperatures,indent=2)+'\n')
    test=collected['test_v5_new']
    report={'variant':args.variant,'calibration':fit,'test':{
        'n':len(test),'accuracy':float(np.mean([np.argmax(r['logits'])==np.argmax(r['target']) for r in test])),
        'raw_nll':float(np.mean([nll(r['logits'],r['target'],1.) for r in test])),
        'calibrated_nll':float(np.mean([nll(r['logits'],r['target'],temperatures[r['type']]) for r in test]))},
        'data_sha256':{s:hashlib.sha256((ROOT/'data'/f'{s}.jsonl').read_bytes()).hexdigest() for s in collected},
        'seconds':time.monotonic()-started}
    (out/'calibration-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
