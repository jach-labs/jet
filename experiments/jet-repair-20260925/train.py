"""Matched short trials selected using per-task metrics, never final test rows."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'src'))
from qwen35_training import load_model,examples,label_logits,decision_loss
FAMILIES=['sarcasm','stance','code','preference','relevance']
def family(source):
 return next((f for f in FAMILIES if f in source),'retention')
def read_data(path,tok):
 rows=[json.loads(l) for l in path.open()];data=examples(path,tok,3072)
 for row,ex in zip(rows,data):
  ex['family']=family(row['source'])
  ex['semantic_labels']=list(row['question'].get('criteria',{}).values()) if ex['type']=='choice' else list(range(len(ex['target'])))
 return data

def f1(gold,pred,label):
 tp=sum(a==label and b==label for a,b in zip(gold,pred));fp=sum(a!=label and b==label for a,b in zip(gold,pred));fn=sum(a==label and b!=label for a,b in zip(gold,pred))
 return 2*tp/max(1,2*tp+fp+fn)
@torch.no_grad()
def evaluate(model,data):
 was=model.training;model.eval();groups=defaultdict(list)
 for ex in data:
  z=label_logits(model,ex).double();logp=z.log_softmax(-1);target=torch.tensor(ex['target'],device=z.device)
  gi=int(target.argmax());pi=int(z.argmax());labels=ex['semantic_labels'];groups[ex['family']].append((float(-(target*logp).sum()),gi==pi,labels[gi],labels[pi]))
 model.train(was);scores={}
 for key,rows in groups.items():
  g=[r[2] for r in rows];p=[r[3] for r in rows];acc=float(np.mean([r[1] for r in rows]));nll=float(np.mean([r[0] for r in rows]))
  metric=f1(g,p,'sarcastic') if key=='sarcasm' else float(np.mean([f1(g,p,c) for c in sorted(set(g)|set(p))])) if key in ['stance','relevance'] else acc
  scores[key]={'n':len(rows),'accuracy':acc,'nll':nll,'metric':metric,'metric_name':'positive-class F1' if key=='sarcasm' else 'macro-F1' if key in ['stance','relevance'] else 'accuracy'}
 focus=np.mean([scores[k]['metric'] for k in FAMILIES if k in scores])
 objective=.7*float(focus)+.3*scores['retention']['accuracy'] if 'retention' in scores else float(focus)
 return {'objective':objective,'families':scores}
def eligible(scores,baseline):
 return (scores['families']['sarcasm']['metric']>=baseline['families']['sarcasm']['metric']-.02 and scores['families']['retention']['accuracy']>=baseline['families']['retention']['accuracy']-.02)
def main():
 p=argparse.ArgumentParser();p.add_argument('--lr',type=float,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 assert not args.out.exists(),args.out;args.out.mkdir(parents=True)
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.88);seed=250925
 torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 model,tok,info=load_model(str(ROOT/'releases/jet-v6'),'e5b8f610ddb92ffaba596ae452bed32a9fef49ca',train=True,rank=16)
 train=read_data(HERE/'train.jsonl',tok)
 if args.smoke:
  train=sorted(train,key=lambda e:len(e['ids']));train=[train[len(train)//2],train[int(len(train)*.9)],train[-1]];val=[]
 else:val=read_data(HERE/'selection.jsonl',tok)
 order=list(train);random.Random(seed).shuffle(order);accum=1 if args.smoke else 4;steps=min(2000,math.ceil(len(order)/accum));warmup=min(100,max(1,steps//10))
 params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
 config={'lr':args.lr,'steps':steps,'seed':seed,'accumulate':accum,'rank':16,'base_revision':'e5b8f610ddb92ffaba596ae452bed32a9fef49ca','train_sha256':hashlib.sha256((HERE/'train.jsonl').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((HERE/'selection.jsonl').read_bytes()).hexdigest(),'selection':'70% mean task metrics + 30% retention accuracy; sarcasm F1 and retention accuracy each must not fall more than 2pp below initial. Step0 fallback.','loading_info':info}
 (args.out/'config.json').write_text(json.dumps(config,indent=2,default=lambda value: sorted(value) if isinstance(value,set) else str(value))+'\n')
 if not args.smoke:
  baseline=evaluate(model,val);best=baseline;best_step=0
  model.save_pretrained(args.out/'best');tok.save_pretrained(args.out/'best')
  (args.out/'best-step.json').write_text(json.dumps({'step':0,**best},indent=2)+'\n')
  (args.out/'metrics.jsonl').write_text(json.dumps({'step':0,**baseline})+'\n');print('BASELINE',json.dumps(baseline),flush=True)
 started=time.monotonic()
 for step in range(1,steps+1):
  batch=order[(step-1)*accum:step*accum];optimizer.zero_grad(set_to_none=True)
  lr=args.lr*min(1.,step/warmup)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-1-warmup)/max(1,steps-warmup))))
  for group in optimizer.param_groups:group['lr']=lr
  loss_sum=0
  for ex in batch:
   z=label_logits(model,ex);loss=decision_loss(z,torch.tensor(ex['target']),ex['type']=='score',1.)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   (loss/len(batch)).backward();loss_sum+=loss.item()/len(batch)
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
  if step%10==0 or args.smoke:print(json.dumps({'step':step,'total':steps,'loss':loss_sum,'lr':lr,'grad_norm':float(norm),'elapsed_seconds':time.monotonic()-started,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9}),flush=True)
  if not args.smoke and (step%500==0 or step==steps):
   scores=evaluate(model,val);allowed=eligible(scores,baseline);improved=allowed and scores['objective']>best['objective']
   with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,'eligible':allowed,'improved':improved,**scores})+'\n')
   model.save_pretrained(args.out/'last');tok.save_pretrained(args.out/'last')
   torch.save({'step':step,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},args.out/'optimizer.pt')
   if improved:
    best=scores;best_step=step;model.save_pretrained(args.out/'best');tok.save_pretrained(args.out/'best');(args.out/'best-step.json').write_text(json.dumps({'step':step,**best},indent=2)+'\n')
   print('VALIDATION',step,json.dumps(scores),'eligible',allowed,'improved',improved,flush=True)
 done={'steps':steps,'best_step':best_step if not args.smoke else None,'seconds':time.monotonic()-started,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9}
 (args.out/'completed.json').write_text(json.dumps(done,indent=2)+'\n');print('COMPLETE',json.dumps(done),flush=True)
if __name__=='__main__':main()
