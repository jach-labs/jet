"""Exercise real training lengths, finite gradients, and adapter serialization."""
import json,time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint
from train import load_examples,to_arrays,loss_fn,save_adapter

BASE='/home/jach/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-bf16/snapshots/7aef04e9adfd926ce0da9da376fe9610c8818a58'
mx.set_cache_limit(512*1024**2)
mx.set_memory_limit(12*1024**3)
model,tok=load(BASE); model.freeze()
p={'rank':16,'scale':10.,'dropout':.05};linear_to_lora_layers(model,len(model.layers),p)
grad_checkpoint(model.layers[0]);model.train()
examples=load_examples(Path('data/train_v5_r2.jsonl'),tok,2048,.02,True)
examples.sort(key=lambda x:len(x.tokens))
optimizer=optim.AdamW(learning_rate=1e-6)
vg=nn.value_and_grad(model,loss_fn)
report=[]
for index in [len(examples)//2,int(len(examples)*.9),len(examples)-1]:
 x=examples[index];start=time.monotonic(); loss,grads=vg(model,*to_arrays([x]))
 optimizer.update(model,grads);mx.eval(model.trainable_parameters(),optimizer.state,loss)
 finite=all(bool(mx.all(mx.isfinite(v))) for _,v in tree_flatten(grads))
 r={'tokens':len(x.tokens),'loss':loss.item(),'gradients_finite':finite,'seconds':time.monotonic()-start,'peak_gb':mx.get_peak_memory()/1e9};print(json.dumps(r),flush=True);report.append(r)
 assert finite and bool(mx.isfinite(loss))
config={'fine_tune_type':'lora','num_layers':len(model.layers),'lora_parameters':p,'base_model':'mlx-community/Qwen3.5-0.8B-MLX-bf16','base_revision':'7aef04e9adfd926ce0da9da376fe9610c8818a58'}
save_adapter(model,Path('adapters/jet-08b-capacity-smoke'),config,optimizer)
Path('docs/training/jet-08b/capacity-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
