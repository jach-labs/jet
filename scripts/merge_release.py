"""Merge a Qwen3.5 PEFT LoRA adapter into complete bf16 text-backbone weights.

    uv run python scripts/merge_release.py --adapter adapters/<run>/best --step 3750

Writes sharded weights, config, tokenizer, calibration and merge-provenance.json
into the release folder (default releases/jet-v6), next to its tracked runtime files.
Memory-bounded: tensors are merged one at a time and written in ~1 GB shards.
Generalized from experiments/jet-4b-full-20260924/merge_release.py (the v6 merge).
"""
import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig

from release_files import DEFAULT_RELEASE, sync_code

SHARD_BYTES = 1_000_000_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='Qwen/Qwen3.5-4B')
    ap.add_argument('--base-revision', default='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    ap.add_argument('--adapter', type=Path, required=True, help='PEFT adapter dir with calibration.json and tokenizer files')
    ap.add_argument('--step', type=int, required=True, help='training step of the adapter, recorded in provenance')
    ap.add_argument('--out', type=Path, default=DEFAULT_RELEASE)
    args = ap.parse_args()

    torch.set_num_threads(2)
    base = Path(snapshot_download(args.base, revision=args.base_revision))
    cfg = json.loads((args.adapter / 'adapter_config.json').read_text())
    if cfg.get('use_dora') or cfg.get('use_rslora') or cfg.get('fan_in_fan_out'):
        raise SystemExit('Only plain LoRA adapters are supported')
    scale = cfg['lora_alpha'] / cfg['r']

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in [*args.out.glob('model-*.safetensors'), *args.out.glob('part-*.safetensors')]:
        stale.unlink()
    config = Qwen3_5TextConfig(**json.loads((base / 'config.json').read_text())['text_config'])
    config.architectures = ['Qwen3_5ForCausalLM']
    config.save_pretrained(args.out)

    shards, batch, size, total, mapping, merged = [], {}, 0, 0, {}, []

    def flush():
        nonlocal batch, size
        if not batch:
            return
        name = f'part-{len(shards) + 1:05d}.safetensors'
        save_file(batch, args.out / name, metadata={'format': 'pt'})
        mapping.update(dict.fromkeys(batch, name))
        shards.append(name)
        batch, size = {}, 0
        gc.collect()

    with safe_open(args.adapter / 'adapter_model.safetensors', framework='pt', device='cpu') as adapter:
        used = set()
        for path in sorted(base.glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as weights:
                for key in weights.keys():
                    # Keep the text backbone only; the vision tower is unused.
                    if not key.startswith('model.language_model.'):
                        continue
                    dest = key.replace('model.language_model.', 'model.', 1)
                    w = weights.get_tensor(key).clone()
                    stem = 'base_model.model.' + dest.removesuffix('.weight')
                    a_key, b_key = stem + '.lora_A.weight', stem + '.lora_B.weight'
                    if a_key in adapter.keys():
                        # PEFT merge arithmetic: fp32 B @ A * alpha / r, added to bf16 and rounded.
                        w.add_((adapter.get_tensor(b_key).float() @ adapter.get_tensor(a_key).float()) * scale)
                        if not torch.isfinite(w).all():
                            raise RuntimeError(f'non-finite merged weight: {dest}')
                        used.update([a_key, b_key])
                        merged.append(dest)
                    n = w.numel() * w.element_size()
                    if size + n > SHARD_BYTES:
                        flush()
                    batch[dest] = w
                    size += n
                    total += n
        if unused := set(adapter.keys()) - used:
            raise RuntimeError(f'adapter tensors not merged: {sorted(unused)[:5]}')
    flush()

    for i, old in enumerate(shards, 1):
        new = f'model-{i:05d}-of-{len(shards):05d}.safetensors'
        (args.out / old).rename(args.out / new)
        mapping = {k: new if v == old else v for k, v in mapping.items()}
    (args.out / 'model.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': total}, 'weight_map': mapping}, indent=2) + '\n')
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'calibration.json']:
        shutil.copy2(args.adapter / name, args.out / name)
    sync_code(args.out)
    provenance = {
        'base': args.base, 'base_revision': args.base_revision,
        'adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'step': args.step,
        'merge': 'BF16 backbone plus FP32 B@A times alpha/r, rounded to BF16 (PEFT default merge arithmetic)',
        'architecture': 'complete Qwen3_5ForCausalLM text backbone; tied lm_head; unused vision tower omitted',
        'merged_modules': len(merged), 'tensors': len(mapping), 'bytes': total,
    }
    (args.out / 'merge-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Merged {len(merged)} modules; {len(mapping)} tensors; {total} bytes into {args.out}', flush=True)


if __name__ == '__main__':
    main()
