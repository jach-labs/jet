"""Build a 4-bit browser candidate from original bf16 weights and the released ONNX graph.

The embedding stays at int8; all MatMulNBits weights are rebuilt from bf16 (never
requantized from int8). Requires onnx, onnxruntime, safetensors, torch, numpy.
"""
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--block-size', type=int, default=32)
    args = ap.parse_args()
    model = onnx.load(str(args.source / 'onnx/model_q8.onnx'))
    replacements = {}
    block = args.block_size
    with safe_open(str(args.source / 'model.safetensors'), framework='pt') as weights:
        for i, node in enumerate(n for n in model.graph.node if n.op_type == 'MatMulNBits'):
            name = node.name.removesuffix('/MatMul_q').strip('/').replace('/', '.') + '.weight'
            if name == 'lm_head.weight': name = 'model.embed_tokens.weight'
            w = np.ascontiguousarray(weights.get_tensor(name).float().numpy().T)
            k, n = w.shape
            blocks = (k + block - 1) // block
            if k % block: w = np.pad(w, ((0, blocks * block - k), (0, 0)))
            packed = np.zeros((n, blocks, block // 2), dtype=np.uint8)
            scales = np.zeros((n, blocks), dtype=np.float32)
            zeros = np.zeros((n, (blocks + 1) // 2), dtype=np.uint8)
            quantize_matmul_4bits(packed, w, scales, zeros, block, n, k, True)
            replacements[node.input[1]] = numpy_helper.from_array(packed, node.input[1])
            replacements[node.input[2]] = numpy_helper.from_array(scales, node.input[2])
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute(key, value) for key, value in
                                   dict(K=k, N=n, bits=4, block_size=block).items()])
            if i % 28 == 0: print(f'{i + 1}/197 {name}', flush=True)
    initializers = [replacements.get(t.name, t) for t in model.graph.initializer]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f'Saved {args.output}: {args.output.stat().st_size:,} bytes', flush=True)

if __name__ == '__main__': main()
