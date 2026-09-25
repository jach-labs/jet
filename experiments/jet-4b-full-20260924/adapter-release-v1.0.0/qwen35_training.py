"""BF16 Qwen3.5 text-only LoRA backend for Jet's label-distribution objective."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from format import Question, label_token_ids
from inference import encode

TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj']

def install_linear_attention():
    """Use FLA's differentiable chunk kernel; leave convolution on native PyTorch."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as impl
    def chunk(query, key, value, g, beta, chunk_size=64, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        return chunk_gated_delta_rule(q=query,k=key,v=value,g=g,beta=beta,
            initial_state=initial_state,output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    impl.torch_chunk_gated_delta_rule = chunk


def load_model(base, revision=None, adapter=None, train=False, rank=16):
    install_linear_attention()
    tok=AutoTokenizer.from_pretrained(base,revision=revision)
    model, info=Qwen3_5ForCausalLM.from_pretrained(base,revision=revision,
        dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa',
        output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise RuntimeError(f'Incomplete backbone load: {info}')
    model.config.use_cache=False
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=train)
    elif train:
        model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=.05,
            target_modules=TARGET_MODULES,bias='none',task_type='CAUSAL_LM'))
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.train()
    else:model.eval()
    return model,tok,info


def examples(path, tok, max_prompt=None):
    result=[]
    for line in Path(path).open():
        row=json.loads(line);q=Question.from_dict(row['question'])
        ids=encode(tok,row['state'],q,10**9) # No state truncation.
        if max_prompt is not None and len(ids)>max_prompt:
            raise ValueError(f'Prompt with {len(ids)} tokens exceeds {max_prompt}: {row["source"]}')
        result.append({'ids':ids,'labels':label_token_ids(tok,q),'target':row['target'],
                       'type':q.type,'source':row['source'],'weight':float(row.get('weight',1.))})
    return result


def label_logits(model, example):
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    ids=torch.tensor([example['ids']],device='cuda',dtype=torch.long)
    hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state[:, -1, :]
    labels=torch.tensor(example['labels'],device='cuda',dtype=torch.long)
    # Only materialize the requested rows of the frozen vocabulary head.
    return F.linear(hidden,base.lm_head.weight.index_select(0,labels))[0].float()


def decision_loss(logits,target,ordinal=False,weight=1.,smoothing=.02,ordinal_weight=2.):
    target=target.to(logits.device,dtype=torch.float32)
    target=(1-smoothing)*target+smoothing/len(target)
    logp=F.log_softmax(logits,dim=-1)
    loss=-(target*logp).sum()
    if ordinal:
        # Same normalized ranked-probability score as the MLX trainer.
        loss=loss+ordinal_weight*((logp.exp().cumsum(-1)[:-1]-target.cumsum(-1)[:-1])**2).mean()
    return loss*weight


@torch.no_grad()
def evaluate(model, data, temperatures=None, collect=False):
    was_training=model.training;model.eval();losses=[];correct=[];zs=[]
    temperatures=temperatures or {}
    for ex in data:
        z=label_logits(model,ex);p=F.log_softmax(z/temperatures.get(ex['type'],1.),dim=-1)
        t=torch.tensor(ex['target'],device='cuda',dtype=torch.float32)
        losses.append(float(-(t*p).sum()));correct.append(int(z.argmax()==t.argmax()))
        if collect:zs.append(z.cpu().numpy())
    model.train(was_training)
    result={'n':len(data),'nll':float(np.mean(losses)),'acc':float(np.mean(correct))}
    return (result,zs) if collect else result
