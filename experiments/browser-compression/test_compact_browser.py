"""Verify all retained weights and all 267 allowed output logits against q8."""
from pathlib import Path
import argparse,json,numpy as np,onnx
from onnx import numpy_helper,helper
from evaluate_browser_quantization import session,run
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--artifacts',type=Path,default=Path('artifacts/quantization'))
args=ap.parse_args();root=args.artifacts;source=args.source
ids=json.loads((root/'model_q8_compact.labels.json').read_text())['token_ids']
a=onnx.load(str(source/'onnx/model_q8.onnx'));b=onnx.load(str(root/'model_q8_compact.onnx'))
ai={t.name:t for t in a.graph.initializer};bi={t.name:t for t in b.graph.initializer}
head=next(n for n in a.graph.node if n.name=='/lm_head/MatMul_q');attrs={x.name:helper.get_attribute_value(x) for x in head.attribute}
for name,t in ai.items():
 if name in head.input[1:]:
  x=numpy_helper.to_array(t).reshape(attrs['N'],-1)[ids];y=numpy_helper.to_array(bi[name]).reshape(len(ids),-1)
  assert np.array_equal(x,y),name
 else:assert t.SerializeToString()==bi[name].SerializeToString(),name
# Every existing graph node is unchanged except the head's output name/width.
assert len(b.graph.node)==len(a.graph.node)+1
bn={n.name:n for n in b.graph.node}
for n in a.graph.node:
 if n.name!=head.name:assert n.SerializeToString()==bn[n.name].SerializeToString(),n.name
print('PASS: body, embedding and all 267 retained head rows are byte-identical',flush=True)
del a,b,ai,bi
q8=session(source/'onnx/model_q8.onnx',4);compact=session(root/'model_q8_compact.onnx',4)
golden=json.loads((source/'golden.json').read_text());results=[]
for c in golden['cases']:
 z=run(q8,[c['input_ids']])[0,ids];v=run(compact,[c['input_ids']])[0,ids]
 difference=float(np.max(np.abs(z-v)));assert np.isfinite(v).all() and difference<1e-4,(c['name'],difference)
 results.append(dict(name=c['name'],max_logit_difference=difference));print(c['name'],difference,flush=True)
report=dict(weight_equivalence=True,labels_checked=267,cases=len(results),max_logit_difference=max(x['max_logit_difference'] for x in results),results=results)
(root/'compact-equivalence.json').write_text(json.dumps(report,indent=2))
