"""Memory-bounded PEFT LoRA merge into the complete trained text backbone."""
import gc,json,hashlib,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig
BASE=Path('/home/jach/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
ADAPTER=Path('adapters/jet-4b-full-20260924/best')
OUT=Path('releases/jet-v6');OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
cfg=json.loads((ADAPTER/'adapter_config.json').read_text());assert cfg['r']==cfg['lora_alpha']==16 and not cfg['use_dora'] and not cfg['fan_in_fan_out']
config=Qwen3_5TextConfig(**json.loads((BASE/'config.json').read_text())['text_config'])
config.architectures=['Qwen3_5ForCausalLM'];config.save_pretrained(OUT)
shards=[];batch={};size=0;total=0;mapping={};merged=[]
def flush():
 global batch,size
 if not batch:return
 name=f'part-{len(shards)+1:05d}.safetensors';save_file(batch,OUT/name,metadata={'format':'pt'})
 for k in batch:mapping[k]=name
 shards.append(name);batch={};size=0;gc.collect()
with safe_open(ADAPTER/'adapter_model.safetensors',framework='pt',device='cpu') as a:
 used=set()
 for path in sorted(BASE.glob('*.safetensors')):
  with safe_open(path,framework='pt',device='cpu') as b:
   for key in b.keys():
    if not key.startswith('model.language_model.'):continue
    dest=key.replace('model.language_model.','model.',1)
    w=b.get_tensor(key).clone()
    stem='base_model.model.'+dest.removesuffix('.weight')
    ak=stem+'.lora_A.weight';bk=stem+'.lora_B.weight'
    if ak in a.keys():
     delta=a.get_tensor(bk).float() @ a.get_tensor(ak).float()
     w.add_(delta*(cfg['lora_alpha']/cfg['r']))
     assert torch.isfinite(w).all(),dest
     used.update([ak,bk]);merged.append(dest);del delta
    n=w.numel()*w.element_size()
    if size+n>1_000_000_000:flush()
    batch[dest]=w;size+=n;total+=n
 assert used==set(a.keys()),set(a.keys())-used
flush()
for i,old in enumerate(shards,1):
 new=f'model-{i:05d}-of-{len(shards):05d}.safetensors';(OUT/old).rename(OUT/new)
 for k,v in list(mapping.items()):
  if v==old:mapping[k]=new
(OUT/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':mapping},indent=2)+'\n')
for n in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','calibration.json']:shutil.copy2(ADAPTER/n,OUT/n)
shutil.copy2('releases/jet-4b-v1.0.0/LICENSE',OUT/'LICENSE')
(OUT/'merge-provenance.json').write_text(json.dumps({'base':'Qwen/Qwen3.5-4B','base_revision':BASE.name,'adapter_sha256':hashlib.sha256((ADAPTER/'adapter_model.safetensors').read_bytes()).hexdigest(),'step':3750,'merge':'BF16 backbone plus FP32 B@A times alpha/r, rounded to BF16 (PEFT default merge arithmetic)','architecture':'complete Qwen3_5ForCausalLM text backbone; tied lm_head; unused vision tower omitted','merged_modules':len(merged),'tensors':len(mapping),'bytes':total},indent=2)+'\n')
print('Merged',len(merged),'modules;',len(mapping),'tensors;',total,'bytes',flush=True)
