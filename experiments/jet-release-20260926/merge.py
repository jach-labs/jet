import gc,hashlib,json,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'releases/jet-v6.1';ADAPTER=ROOT/'adapters/jet-focused-20260925/lr3e-6/best';OUT=ROOT/'releases/jet-v6.2'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
expected=json.loads((BASE/'release-manifest.json').read_text())['files']
for p in BASE.glob('*.safetensors'):assert sha(p)==expected[p.name]
cfg=json.loads((ADAPTER/'adapter_config.json').read_text());assert cfg['r']==cfg['lora_alpha']==16 and not cfg['use_dora'] and not cfg['fan_in_fan_out']
assert sha(ADAPTER/'adapter_model.safetensors')=='53e01c1a79ec323693c1383a95240d1bf5911cbd13fc44c1e17144f4172998af'
shards=[];batch={};size=0;total=0;mapping={};merged=[]
def flush():
 global batch,size
 if not batch:return
 name=f'part-{len(shards)+1:05d}.safetensors';save_file(batch,OUT/name,metadata={'format':'pt'})
 for k in batch:mapping[k]=name
 shards.append(name);batch={};size=0;gc.collect()
with safe_open(ADAPTER/'adapter_model.safetensors',framework='pt',device='cpu') as a:
 used=set();akeys=set(a.keys())
 for path in sorted(BASE.glob('*.safetensors')):
  with safe_open(path,framework='pt',device='cpu') as b:
   for key in b.keys():
    w=b.get_tensor(key).clone();stem='base_model.model.'+key.removesuffix('.weight');ak=stem+'.lora_A.weight';bk=stem+'.lora_B.weight'
    if ak in akeys:
     delta=a.get_tensor(bk).float() @ a.get_tensor(ak).float();w.add_(delta*(cfg['lora_alpha']/cfg['r']));assert torch.isfinite(w).all(),key
     used.update([ak,bk]);merged.append(key);del delta
    n=w.numel()*w.element_size()
    if size+n>1_000_000_000:flush()
    batch[key]=w;size+=n;total+=n
 assert used==akeys,akeys-used
flush()
for i,old in enumerate(shards,1):
 new=f'model-{i:05d}-of-{len(shards):05d}.safetensors';(OUT/old).rename(OUT/new)
 for k,v in list(mapping.items()):
  if v==old:mapping[k]=new
(OUT/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':mapping},indent=2)+'\n')
for n in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','calibration.json','LICENSE','requirements.txt','jet.py','runtime.py','format.py','inference.py']:shutil.copy2(BASE/n,OUT/n)
(OUT/'merge-provenance.json').write_text(json.dumps({'name':'Jet','version':'v6.2.0','base':'michaljach/jet','base_revision':'f446b82727be57da348bb46eccf211414294ab3e','base_weights_sha256':{k:v for k,v in expected.items() if k.endswith('.safetensors')},'adapter_sha256':sha(ADAPTER/'adapter_model.safetensors'),'step':250,'merge':'Continue from the previously merged Jet v6.1 weights; add FP32 B@A correction and round to BF16. No original-Qwen reset.','merged_modules':len(merged),'tensors':len(mapping),'bytes':total},indent=2)+'\n')
print(json.dumps({'merged_modules':len(merged),'tensors':len(mapping),'bytes':total}),flush=True)

p=OUT/'jet.py';s=p.read_text().replace('8192','16384');p.write_text(s)
