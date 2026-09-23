"""Build the browser model (onnx/model_q8.onnx) from an optimum ONNX export. export_web.sh runs all of it.

    python src/onnx_web.py slice    <export>/model.onnx <export>/model_sliced.onnx
    python src/onnx_web.py quantize <export>/model_sliced.onnx models/jet/onnx/model_q8.onnx
    python src/onnx_web.py check    models/jet/onnx/model_q8.onnx models/jet/golden.json

slice: jet only reads the next-token logits after the last prompt token, so the hidden states are cut to that
position before the LM head (151936 x 1024, as big as the rest of the model). Only the graph is rewritten;
the weights stay in the export's data file.

quantize: MatMul weights become 8-bit com.microsoft MatMulNBits (block 64, ORT's C++ block quantizer), and
the embedding table int8 rows with one fp32 scale each, dequantized after the Gather with standard ops. It
reads one tensor at a time: onnxruntime's own quantizers load the whole 3 GB fp32 model and run out of memory
on a 16 GB machine. Why 8-bit weight-only: 4-bit moved probabilities by up to 0.31 on the golden cases, and
dynamic int8 (MatMulInteger, about 2.6x faster in wasm) flipped 8 of 36 answers, because Qwen3's activation
outliers don't survive per-tensor activation quantization.

check: runs the golden cases (jet-golden) through onnxruntime and compares with the bf16 reference.

Runs in the separate environment export_web.sh sets up (onnx, onnxruntime), not the project's.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CHUNK = 512


def slice_last(src: Path, dst: Path) -> None:
    import onnx
    from onnx import helper, numpy_helper

    assert src.parent == dst.parent, "the sliced graph refers to the export's data file by relative path"
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    head = next(n for n in g.node if "logits" in n.output)
    assert head.op_type == "MatMul", head.op_type
    g.initializer.extend([
        numpy_helper.from_array(np.array([-1], np.int64), "jet_last_start"),
        numpy_helper.from_array(np.array([np.iinfo(np.int64).max], np.int64), "jet_last_end"),
        numpy_helper.from_array(np.array([1], np.int64), "jet_last_axis"),
    ])
    g.node.insert(list(g.node).index(head), helper.make_node(
        "Slice", [head.input[0], "jet_last_start", "jet_last_end", "jet_last_axis"], ["jet_last_hidden"], name="jet_last_slice"))
    head.input[0] = "jet_last_hidden"
    dim = next(o for o in g.output if o.name == "logits").type.tensor_type.shape.dim[1]
    dim.ClearField("dim_param")
    dim.dim_value = 1
    onnx.save(m, dst)
    print(f"{dst}: logits cut to the last position")


def quantize(src: Path, dst: Path, bits: int = 8, block: int = 64) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import load_external_data_for_model
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer

    base = src.parent
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    quantizer = DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits))

    def weight(t) -> np.ndarray:
        # Read straight from the data file: numpy_helper.to_array(base_dir=...) keeps every tensor's bytes on the proto.
        assert t.data_type == TensorProto.FLOAT, t.name
        info = {e.key: e.value for e in t.external_data}
        return np.fromfile(base / info["location"], np.float32, int(np.prod(t.dims)), offset=int(info.get("offset", 0))).reshape(t.dims)

    nodes, new_inits, drop = [], [], set()
    for node in g.node:
        w = inits.get(node.input[1]) if node.op_type == "MatMul" and len(node.input) > 1 else None
        if w is not None and len(w.dims) == 2:
            arr = weight(w)
            packed, scales, _ = quantizer.qbits_block_quant(arr)
            new_inits += [numpy_helper.from_array(packed, w.name + "_q"), numpy_helper.from_array(scales, w.name + "_scales")]
            nodes.append(helper.make_node("MatMulNBits", [node.input[0], w.name + "_q", w.name + "_scales"], list(node.output),
                                          name=node.name + "_q", domain="com.microsoft",
                                          K=arr.shape[0], N=arr.shape[1], bits=bits, block_size=block))
            drop.add(w.name)
            continue
        w = inits.get(node.input[0]) if node.op_type == "Gather" else None
        if w is not None and len(w.dims) == 2:
            arr = weight(w)
            scale = np.abs(arr).max(axis=1, keepdims=True) / 127
            scale[scale == 0] = 1
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            new_inits += [numpy_helper.from_array(q, w.name + "_q"), numpy_helper.from_array(scale.astype(np.float32), w.name + "_scales")]
            out = node.output[0]
            nodes += [
                helper.make_node("Gather", [w.name + "_q", node.input[1]], [out + "_q"], name=node.name + "_q"),
                helper.make_node("Gather", [w.name + "_scales", node.input[1]], [out + "_s"], name=node.name + "_s"),
                helper.make_node("Cast", [out + "_q"], [out + "_f"], name=node.name + "_cast", to=TensorProto.FLOAT),
                helper.make_node("Mul", [out + "_f", out + "_s"], [out], name=node.name + "_deq"),
            ]
            drop.add(w.name)
            continue
        nodes.append(node)

    kept = [t for t in g.initializer if t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(kept + new_inits)
    del g.node[:]
    g.node.extend(nodes)
    load_external_data_for_model(m, str(base))  # the few tensors left unquantized (norm weights and the like)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, dst)
    print(f"{dst}: {dst.stat().st_size / 1e6:.0f} MB, {len(drop)} weights quantized to {bits} bits")


def check(model: Path, golden_path: Path, threads: int = 4) -> bool:
    import onnxruntime as ort

    golden = json.loads(golden_path.read_text())
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    layers = sum(1 for i in sess.get_inputs() if i.name.endswith(".key"))

    def run(ids: list[int]) -> np.ndarray:
        # Chunked through the KV cache, as jet.js does, so a long prompt never materializes n x n attention.
        past = {f"past_key_values.{i}.{kv}": np.zeros((1, 8, 0, 128), np.float32) for i in range(layers) for kv in ("key", "value")}
        for done in range(0, len(ids), CHUNK):
            chunk = ids[done : done + CHUNK]
            out = dict(zip(names, sess.run(None, {
                "input_ids": np.array([chunk], np.int64),
                "attention_mask": np.ones((1, done + len(chunk)), np.int64),
                "position_ids": np.arange(done, done + len(chunk), dtype=np.int64)[None],
                **past,
            })))
            past = {k.replace("present", "past_key_values"): v for k, v in out.items() if k.startswith("present")}
        return out["logits"][0, -1]

    worst_logit = worst_prob = 0.0
    flips, tokens, t0 = [], 0, time.perf_counter()
    for c in golden["cases"]:
        z = run(c["input_ids"])[c["label_token_ids"]]
        tokens += len(c["input_ids"])
        s = z / c["temperature"]
        p = np.exp(s - s.max())
        p /= p.sum()
        worst_logit = max(worst_logit, float(np.abs(z - np.array(c["logits"])).max()))
        worst_prob = max(worst_prob, float(np.abs(p - np.array(c["probabilities"])).max()))
        if int(np.argmax(z)) != int(np.argmax(c["logits"])):
            flips.append(c["name"])
    dt = time.perf_counter() - t0
    print(f"{len(golden['cases'])} golden cases: worst |dlogit| {worst_logit:.3f}, worst |dprob| {worst_prob:.4f}, "
          f"answers flipped {len(flips)} {flips or ''} ({tokens / dt:.0f} tokens/s on {threads} threads)")
    return not flips


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and check the browser (onnxruntime-web) model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("slice", "quantize", "check"):
        p = sub.add_parser(name)
        p.add_argument("src", type=Path)
        p.add_argument("dst", type=Path)
    args = ap.parse_args()
    if args.cmd == "slice":
        slice_last(args.src, args.dst)
    elif args.cmd == "quantize":
        quantize(args.src, args.dst)
    elif not check(args.src, args.dst):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
