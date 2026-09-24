"""Fuse MLX LoRA using its exact scale convention; build CPU reference vectors.

CPU fallback when MLX CUDA cannot initialize. This is not a CUDA parity test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from format import Question
from inference import encode, prompt_text, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, required=True)
    ap.add_argument('--reference-cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.adapter / 'adapter_config.json').read_text())
    weights = load_file(str(args.base / 'model.safetensors'))
    adapter = load_file(str(args.adapter / 'adapters.safetensors'))
    scale = config['lora_parameters']['scale']
    consumed = set()
    count = 0
    for key, a in adapter.items():
        if not key.endswith('.lora_a'):
            continue
        module = key.removesuffix('.lora_a')
        bkey = module + '.lora_b'
        b = adapter[bkey]
        w = weights[module + '.weight']
        assert a.shape[1] == b.shape[0] == config['lora_parameters']['rank']
        delta = ((scale * b.T) @ a.T).to(w.dtype)
        assert delta.shape == w.shape
        weights[module + '.weight'] = w + delta
        consumed.update((key, bkey))
        count += 1
    assert consumed == set(adapter), 'Unconsumed adapter tensors'
    assert count == 28 * 7
    save_file(weights, str(args.out / 'model.safetensors'), metadata={'format': 'pt'})
    del weights, adapter
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'merges.txt', 'vocab.json', 'chat_template.jinja']:
        if (args.base / name).exists():
            shutil.copy2(args.base / name, args.out / name)
    shutil.copy2(args.adapter / 'calibration.json', args.out / 'calibration.json')
    temperatures = json.loads((args.out / 'calibration.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.out, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.out, local_files_only=True, dtype=torch.float32, attn_implementation='sdpa').eval()
    golden = json.loads(args.reference_cases.read_text())
    golden['model'] = 'Jet, CPU fp32 inference of bf16 fused weights'
    golden['temperatures'] = temperatures
    golden['decide'] = []  # No stale answers from the old model.
    with torch.inference_mode():
        for i, case in enumerate(golden['cases']):
            q = Question.from_dict(case['question'])
            ids = encode(tokenizer, case['state'], q, golden['max_state_tokens'])
            assert ids == case['input_ids'], case['name']
            assert prompt_text(tokenizer, case['state'], q, golden['max_state_tokens']) == case['prompt']
            cache = None
            for start in range(0, len(ids), 256):
                output = model.model(input_ids=torch.tensor([ids[start:start+256]]), past_key_values=cache, use_cache=True)
                cache = output.past_key_values
            logits = model.lm_head(output.last_hidden_state[:, -1, :])[0, case['label_token_ids']].float()
            probs = torch.softmax(logits / temperatures[q.type], dim=-1).numpy()
            assert np.isfinite(probs).all()
            case.update(logits=logits.tolist(), probabilities=probs.tolist(), temperature=temperatures[q.type], answer=summarize(q, probs))
            del cache, output
            print(f'{i+1}/{len(golden["cases"])} {case["name"]}', flush=True)
    (args.out / 'golden.json').write_text(json.dumps(golden, ensure_ascii=False, indent=1)+'\n')
    manifest = {'fusion': '(weight + ((scale * lora_b.T) @ lora_a.T).to(weight.dtype)) in PyTorch CPU', 'layers': count, 'adapter_config': config, 'adapter_sha256': hashlib.sha256((args.adapter / 'adapters.safetensors').read_bytes()).hexdigest(), 'reference_backend': 'torch CPU fp32', 'mlx_cuda_parity': 'not checked: CUDA unavailable', 'torch': torch.__version__}
    (args.out / 'export-provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Fused {count} layers and verified {len(golden["cases"])} reference prompts.', flush=True)


if __name__ == '__main__':
    main()
