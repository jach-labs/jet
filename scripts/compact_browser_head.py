"""Remove unused output rows while retaining every supported Jet label exactly.

The input embedding and body are unchanged. Non-label output logits are filled
with the first label's logit for interface compatibility; this export is ONLY
for Jet typed decisions, never general text generation or full-vocabulary scoring.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import onnx
from onnx import helper,numpy_helper
from transformers import AutoTokenizer
from format import choice_labels,MAX_CHOICE_OPTIONS
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
tok=AutoTokenizer.from_pretrained(str(args.source))
labels=choice_labels(tok,MAX_CHOICE_OPTIONS)+list('0123456789')+['no','yes']
ids=[tok.encode(label,add_special_tokens=False)[0] for label in labels]
assert len(set(ids))==267 and all(len(tok.encode(x,add_special_tokens=False))==1 for x in labels)
m=onnx.load(str(args.source/'onnx/model_q8.onnx'))
head=next(n for n in m.graph.node if n.name=='/lm_head/MatMul_q')
attrs={a.name:helper.get_attribute_value(a) for a in head.attribute}
assert attrs['bits']==8 and len(head.input)==3
for t in m.graph.initializer:
 if t.name in head.input[1:]:
  arr=numpy_helper.to_array(t).reshape(attrs['N'],-1)
  if t.name==head.input[1]:arr=arr.reshape(attrs['N'],-1,attrs['block_size'])
  t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr[ids]),t.name))
for a in head.attribute:
 if a.name=='N':a.i=len(ids)
old_output=head.output[0];head.output[0]='jet_label_logits'
# Gather reconstructs the original output shape. Only the 267 documented label
# positions are meaningful and retain their exact original output rows.
lookup=np.zeros(attrs['N'],dtype=np.int64)
for i,token in enumerate(ids):lookup[token]=i
m.graph.initializer.append(numpy_helper.from_array(lookup,'jet_label_lookup'))
index=list(m.graph.node).index(head)
m.graph.node.insert(index+1,helper.make_node('Gather',['jet_label_logits','jet_label_lookup'],[old_output],name='JetTypedLabels',axis=2))
# Old intermediate shape hints may still describe the full-vocabulary head.
for v in m.graph.value_info:
 if v.name=='jet_label_logits':v.type.tensor_type.shape.dim[-1].dim_value=len(ids)
onnx.checker.check_model(m);args.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(args.output))
args.output.with_suffix('.labels.json').write_text(json.dumps(dict(labels=labels,token_ids=ids),indent=2))
print(args.output,args.output.stat().st_size,flush=True)
