"""Decode the ONNX weights exactly for accelerated full-set evaluation on Metal.

This does not create a new quantized model. Every weight comes from the supplied
ONNX candidate; a native ONNX cross-check validates the evaluation path.
"""
import argparse
from pathlib import Path
import onnx
import numpy as np
from onnx import numpy_helper,helper
from safetensors.numpy import save_file
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('model',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
m=onnx.load(str(args.model));init={t.name:numpy_helper.to_array(t) for t in m.graph.initializer};weights={}
for name,arr in init.items():
 if name.startswith('model.') and name.endswith('.weight'):weights[name]=arr.copy()
weights['model.embed_tokens.weight']=init['model.embed_tokens.weight_q'].astype(np.float32)*init['model.embed_tokens.weight_scales']
for n in m.graph.node:
 if n.op_type!='MatMulNBits':continue
 attrs={a.name:helper.get_attribute_value(a) for a in n.attribute};bits=attrs['bits'];block=attrs['block_size'];k=attrs['K']
 packed=init[n.input[1]];scales=init[n.input[2]]
 if bits==4:
  unpacked=np.empty((*packed.shape[:-1],packed.shape[-1]*2),np.float32)
  unpacked[...,0::2]=packed&15;unpacked[...,1::2]=packed>>4
 else:unpacked=packed.astype(np.float32)
 w=((unpacked-2**(bits-1))*scales.reshape(*packed.shape[:2],1)).reshape(attrs['N'],-1)[:,:k]
 name=n.name.removesuffix('/MatMul_q').strip('/').replace('/','.')+'.weight'
 weights[name]=np.ascontiguousarray(w)
args.output.parent.mkdir(parents=True,exist_ok=True)
save_file(weights,str(args.output));print(args.output,len(weights),flush=True)
