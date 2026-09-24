import json
from pathlib import Path
import mlx.core as mx
from huggingface_hub import snapshot_download
mx.set_default_device(mx.cpu)
base=Path(snapshot_download('mlx-community/Qwen3-0.6B-bf16',revision='42096995f6402fde107068cf530136fe64b604f8'))
b=mx.load(str(base/'model.safetensors')); a=mx.load('adapters/jet/adapters.safetensors'); pub=mx.load('models/jet/model.safetensors')
report={'keys_match':b.keys()==pub.keys(),'exact_tensors':0,'different_tensors':0,'max_abs_difference':0.0}
for key,weight in b.items():
 prefix=key.removesuffix('.weight')
 if prefix+'.lora_a' in a:
  delta=((10*a[prefix+'.lora_b'].T)@a[prefix+'.lora_a'].T).astype(weight.dtype)
  weight=weight+delta
 diff=float(mx.max(mx.abs(weight.astype(mx.float32)-pub[key].astype(mx.float32))).item())
 report['exact_tensors' if diff==0 else 'different_tensors']+=1
 report['max_abs_difference']=max(report['max_abs_difference'],diff)
Path('docs/training/jet-next/baseline-adapter-check.json').write_text(json.dumps(report,indent=2))
print(report)
