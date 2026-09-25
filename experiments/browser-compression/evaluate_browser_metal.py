"""Evaluate exactly decoded ONNX weights on Metal, cross-checked with native ONNX."""
import argparse,json,time,sys,gc,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3 import Model,ModelArgs
from transformers import AutoTokenizer
from model import pad_batch,pad_labels,label_logits
from format import Question,label_token_ids
from inference import encode
from evaluate import metrics
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--artifacts',type=Path,default=Path('artifacts/quantization'))
args=ap.parse_args();root=args.artifacts;source=args.source
rows=[json.loads(l) for l in Path('data/test.jsonl').read_text().splitlines()]
tok=AutoTokenizer.from_pretrained(str(source));temps=json.loads((source/'calibration.json').read_text())
items=[]
for i,r in enumerate(rows):
 q=Question.from_dict(r['question']);ids=encode(tok,r['state'],q,2048);items.append((i,ids,label_token_ids(tok,q)))
items.sort(key=lambda x:len(x[1]));mx.set_cache_limit(500_000_000)
for key in ['q8','q4']:
 path=root/f'{key}-metal.jsonl'
 manifest=dict(weights_sha256=hashlib.sha256((root/f'{key}-eval.safetensors').read_bytes()).hexdigest(),data_sha256=hashlib.sha256(Path('data/test.jsonl').read_bytes()).hexdigest(),max_state_tokens=2048,calibration=temps)
 meta=path.with_suffix('.manifest.json')
 if meta.exists() and json.loads(meta.read_text())!=manifest:raise ValueError('Results belong to a different experiment')
 meta.write_text(json.dumps(manifest,indent=2))
 done={json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
 pending=[x for x in items if x[0] not in done]
 config=json.loads((source/'config.json').read_text());config['tie_word_embeddings']=False
 model=Model(ModelArgs.from_dict(config));model.load_weights(str(root/f'{key}-eval.safetensors'));model.eval();mx.eval(model.parameters())
 begun=time.perf_counter()
 with path.open('a',buffering=1) as f:
  start=0
  while start<len(pending):
   size=min(16,max(1,8192//len(pending[start][1])))
   batch=pending[start:start+size];start+=len(batch)
   tokens,lengths=pad_batch([x[1] for x in batch]);labels,mask=pad_labels([x[2] for x in batch])
   logits=np.array(label_logits(model,tokens,lengths,labels,mask))
   for j,(i,_,ls) in enumerate(batch):
    z=logits[j,:len(ls)]/temps[rows[i]['question']['type']];p=np.exp(z-z.max());p/=p.sum()
    f.write(json.dumps(dict(index=i,probabilities=p.tolist()))+'\n')
   print(f'{key} {len(done)+start}/{len(rows)} {time.perf_counter()-begun:.1f}s',flush=True)
 del model;gc.collect();mx.clear_cache()
records={key:{r['index']:r['probabilities'] for r in map(json.loads,(root/f'{key}-metal.jsonl').read_text().splitlines())} for key in ['q8','q4']}
report={'evaluation_backend':'MLX Metal, float32 weights decoded exactly from each ONNX file; native ONNX cross-check below','n':len(rows),'max_state_tokens':2048,'calibration':temps,'data_sha256':hashlib.sha256(Path('data/test.jsonl').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((root/'model_q4.onnx').read_bytes()).hexdigest()}
targets=[np.array(r['target']) for r in rows]
for key in records:report[key]=metrics([np.array(records[key][i]) for i in range(len(rows))],targets)
report['agreement']=float(np.mean([np.argmax(records['q8'][i])==np.argmax(records['q4'][i]) for i in range(len(rows))]))
report['by_type']={};report['by_source']={}
for field in ['type','source']:
 values={r['question']['type'] if field=='type' else r['source'] for r in rows}
 for val in sorted(values):
  ix=[i for i,r in enumerate(rows) if (r['question']['type'] if field=='type' else r['source'])==val]
  report['by_'+field][val]={key:metrics([np.array(records[key][i]) for i in ix],[targets[i] for i in ix]) for key in records}
native=[json.loads(l) for l in (root/'comparison.jsonl').read_text().splitlines()]
report['native_onnx_crosscheck']={}
for key in records:
 diffs=[float(np.max(np.abs(np.array(r[key])-records[key][r['index']]))) for r in native]
 flips=sum(np.argmax(r[key])!=np.argmax(records[key][r['index']]) for r in native)
 report['native_onnx_crosscheck'][key]=dict(n=len(native),max_probability_difference=max(diffs),argmax_mismatches=int(flips))
report['gates']={'accuracy_drop_max':.005,'nll_increase_max':.02,'ece_increase_max':.01}
report['crosscheck_pass']=all(v['argmax_mismatches']==0 and v['max_probability_difference']<.001 for v in report['native_onnx_crosscheck'].values())
report['quality_pass']=bool(report['crosscheck_pass'] and report['q8']['acc']-report['q4']['acc']<=.005 and report['q4']['nll']-report['q8']['nll']<=.02 and report['q4']['ece']-report['q8']['ece']<=.01)
(root/'quality-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['by_type','by_source']},indent=2))
