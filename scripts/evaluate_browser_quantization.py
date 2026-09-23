"""Paired q8/q4 evaluation on identical browser prompts; saves resumable per-row results."""
import argparse, hashlib, json, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from format import Question, label_token_ids
from inference import encode


def session(path, threads):
    opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])


def run(s, tokens):
    batch=len(tokens); length=max(map(len,tokens))
    ids=np.zeros((batch,length),np.int64); mask=np.zeros_like(ids); positions=np.zeros_like(ids)
    for i,row in enumerate(tokens):
        ids[i,-len(row):]=row;mask[i,-len(row):]=1;positions[i,-len(row):]=np.arange(len(row))
    cache={x.name:np.zeros((batch,8,0,128),np.float32) for x in s.get_inputs() if x.name.startswith('past_key_values.')}
    outputs_names=[x.name for x in s.get_outputs()]
    for start in range(0,length,128):
        end=min(start+128,length)
        out=s.run(None,dict(input_ids=ids[:,start:end],attention_mask=mask[:,:end],position_ids=positions[:,start:end],**cache))
        cache={name.replace('present.','past_key_values.'):v for name,v in zip(outputs_names[1:],out[1:])}
    return out[0][:,-1,:]


def metrics(rows, key):
    ps=[np.array(r[key]) for r in rows];ts=[np.array(r['target']) for r in rows]
    conf=np.array([p.max() for p in ps]);hit=np.array([p.argmax()==t.argmax() for p,t in zip(ps,ts)])
    ece=0
    for low in np.linspace(0,1,10,endpoint=False):
        sel=(conf>low)&(conf<=low+.1)
        if sel.any():ece+=sel.mean()*abs(conf[sel].mean()-hit[sel].mean())
    return dict(n=len(rows),accuracy=float(hit.mean()),nll=float(np.mean([-sum(t*np.log(np.clip(p,1e-12,1))) for p,t in zip(ps,ts)])),ece=float(ece),brier=float(np.mean([sum((p-t)**2) for p,t in zip(ps,ts)])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('data/test.jsonl'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(str(args.source))
    temps=json.loads((args.source/'calibration.json').read_text())
    rows=[json.loads(l) for l in args.data.read_text().splitlines()]
    items=[]
    for i,r in enumerate(rows):
        q=Question.from_dict(r['question']);tokens=encode(tok,r['state'],q,max_state_tokens=2048)
        if len(tokens)>4096:raise ValueError(f'Row {i} exceeds browser limit')
        items.append((i,tokens,label_token_ids(tok,q)))
    items.sort(key=lambda x:len(x[1]))
    if args.limit:items=items[:args.limit]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),max_state_tokens=2048,temperatures=temps,threads=args.threads,batch_size=args.batch_size)
    meta=args.output.with_suffix('.manifest.json')
    if meta.exists() and json.loads(meta.read_text())!=manifest:raise ValueError('Output belongs to a different experiment')
    meta.write_text(json.dumps(manifest,indent=2))
    records=[json.loads(l) for l in args.output.read_text().splitlines()] if args.output.exists() else []
    done={r['index'] for r in records};items=[x for x in items if x[0] not in done]
    sessions={'q8':session(args.source/'onnx/model_q8.onnx',args.threads),'q4':session(args.candidate,args.threads)}
    begun=time.perf_counter()
    with args.output.open('a',buffering=1) as dest:
        for start in range(0,len(items),args.batch_size):
            batch=items[start:start+args.batch_size];result={};timings={}
            for key,s in sessions.items():
                t=time.perf_counter();out=run(s,[x[1] for x in batch]);timings[key]=(time.perf_counter()-t)*1000/len(batch)
                result[key]=[]
                for j,(ix,tokens,labels) in enumerate(batch):
                    z=out[j,labels]/temps[rows[ix]['question']['type']];p=np.exp(z-z.max());p/=p.sum();result[key].append(p.tolist())
            for j,(ix,tokens,_) in enumerate(batch):
                r=dict(index=ix,source=rows[ix]['source'],type=rows[ix]['question']['type'],target=rows[ix]['target'],tokens=len(tokens),q8=result['q8'][j],q4=result['q4'][j],ms=timings)
                records.append(r);dest.write(json.dumps(r)+'\n')
            print(f'{len(records)}/{len(rows)} evaluated; {time.perf_counter()-begun:.1f}s; batch ms/question {timings}',flush=True)
    report=dict(manifest=manifest,q8=metrics(records,'q8'),q4=metrics(records,'q4'),agreement=float(np.mean([np.argmax(r['q8'])==np.argmax(r['q4']) for r in records])),by_type={},by_source={})
    for field in ['type','source']:
        for value in sorted({r[field] for r in records}):
            group=[r for r in records if r[field]==value];report['by_'+field][value]={key:metrics(group,key) for key in sessions}
    report['gates']={'accuracy_drop_max':.005,'nll_increase_max':.02,'ece_increase_max':.01}
    report['quality_pass']=report['q8']['accuracy']-report['q4']['accuracy']<=.005 and report['q4']['nll']-report['q8']['nll']<=.02 and report['q4']['ece']-report['q8']['ece']<=.01
    args.output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['by_type','by_source','manifest']},indent=2))

if __name__=='__main__':main()
