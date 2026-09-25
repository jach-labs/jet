"""Train a fresh BF16 Qwen3.5 LoRA adapter without changing published Jet."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from qwen35_training import load_model,examples,label_logits,decision_loss,evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='Qwen/Qwen3.5-4B');p.add_argument('--revision',required=True)
    p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--rank',type=int,default=16)
    p.add_argument('--accumulate',type=int,default=8);p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--max-prompt',type=int,default=3072);p.add_argument('--seed',type=int,default=240928)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();assert not args.out.exists(),args.out
    args.out.mkdir(parents=True);torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.88)
    config=vars(args).copy();config['out']=str(args.out)
    config['data_sha256']={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [args.train,args.val]}
    config.update(dtype='bfloat16',quantization=None,gradient_checkpointing=True,microbatch=1)
    (args.out/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    model,tok,info=load_model(args.base,args.revision,train=True,rank=args.rank)
    frozen={str(x.dtype) for x in model.parameters() if not x.requires_grad}
    assert frozen=={'torch.bfloat16'},frozen
    train=examples(args.train,tok,args.max_prompt);val=examples(args.val,tok)
    if args.smoke:
        ordered=sorted(train,key=lambda x:len(x['ids']))
        train=[ordered[len(ordered)//2],ordered[int(.9*len(ordered))],ordered[-1]]
        val=train;args.epochs=1;args.accumulate=1;args.eval_every=3
    params=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    steps=math.ceil(len(train)/args.accumulate)*args.epochs
    warmup=max(1,min(100,steps//10));step=0;started=time.monotonic()
    config.update(train_examples=len(train),validation_examples=len(val),total_steps=steps,
                  trainable_parameters=sum(x.numel() for x in params),frozen_dtypes=sorted(frozen),loading_info=info)
    (args.out/'training_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    print(json.dumps(config,default=str),flush=True)
    best=evaluate(model,val);history=args.out/'metrics.jsonl'
    history.write_text(json.dumps({'step':0,'initial':True,**best})+'\n')
    model.save_pretrained(args.out/'best');tok.save_pretrained(args.out/'best')
    print('Initial validation',best,flush=True)
    for epoch in range(args.epochs):
        order=list(train);random.Random(args.seed+epoch).shuffle(order)
        for start in range(0,len(order),args.accumulate):
            batch=order[start:start+args.accumulate];optimizer.zero_grad(set_to_none=True)
            lr=args.lr*min(1.,(step+1)/warmup)*(.05+.95*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,steps-warmup))))
            for group in optimizer.param_groups:group['lr']=lr
            total_loss=0.;t0=time.monotonic()
            for ex in batch:
                z=label_logits(model,ex);loss=decision_loss(z,torch.tensor(ex['target']),ex['type']=='score',ex['weight'])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                (loss/len(batch)).backward();total_loss+=loss.item()/len(batch)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            if args.smoke or step%10==0:
                print(json.dumps({'step':step,'of':steps,'loss':total_loss,'lr':lr,'grad_norm':float(norm),'step_seconds':time.monotonic()-t0,'elapsed_seconds':time.monotonic()-started,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'tokens':sum(len(x['ids']) for x in batch)}),flush=True)
            if step%args.eval_every==0 or step==steps:
                scores=evaluate(model,val);improved=scores['nll']<best['nll']
                with history.open('a') as f:f.write(json.dumps({'step':step,'improved':improved,**scores})+'\n')
                print('Validation',step,scores,improved,flush=True)
                model.save_pretrained(args.out/'last');tok.save_pretrained(args.out/'last')
                torch.save({'step':step,'epoch':epoch,'optimizer':optimizer.state_dict()},args.out/'optimizer.pt')
                if improved:
                    best=scores;model.save_pretrained(args.out/'best');tok.save_pretrained(args.out/'best')
                    (args.out/'best-step.json').write_text(json.dumps({'step':step,**best})+'\n')
    (args.out/'completed.json').write_text(json.dumps({'steps':step,'seconds':time.monotonic()-started,'best':best,'peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9},indent=2)+'\n')
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
