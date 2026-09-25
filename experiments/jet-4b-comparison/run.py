"""Controlled, local BF16 LoRA pilot; run with .venv-qwen35/bin/python."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from qwen35_training import load_model, examples, label_logits, decision_loss
from format import Question, label_token_ids
from inference import encode

SEED = 240924
MAX_PROMPT = 2048
TRAIN_PER_SOURCE = 20
VAL_PER_SOURCE = 10
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    models = json.loads((HERE / 'models.json').read_text())
    tokenizers = [AutoTokenizer.from_pretrained(m['repo'], revision=m['revision']) for m in models]
    report = {'seed': SEED, 'max_prompt': MAX_PROMPT, 'splits': {}}
    split_states = []
    for name, source, count in [('train', 'train_v5_r2', TRAIN_PER_SOURCE), ('validation', 'selection_v5', VAL_PER_SOURCE)]:
        path = ROOT / 'data' / (source + '.jsonl')
        rows = [json.loads(line) for line in path.open()]
        groups = defaultdict(list)
        for row in rows:
            groups[row['source']].append(row)
        chosen, excluded, lengths = [], Counter(), []
        rng = random.Random(SEED)
        for family in sorted(groups):
            rng.shuffle(groups[family])
            selected = []
            for row in groups[family]:
                q = Question.from_dict(row['question'])
                sizes = [len(encode(tok, row['state'], q, 10**9)) for tok in tokenizers]
                if max(sizes) > MAX_PROMPT:
                    excluded[family] += 1
                    continue
                for tok in tokenizers:
                    assert len(label_token_ids(tok, q)) == len(row['target'])
                selected.append(row)
                lengths.append(sizes)
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f'Insufficient complete prompts for {family}: {len(selected)}')
            chosen.extend(selected)
        rng.shuffle(chosen)
        output = HERE / (name + '.jsonl')
        output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in chosen))
        split_states.append({json.dumps(row['state'], sort_keys=True) for row in chosen})
        report['splits'][name] = {'input': str(path), 'input_sha256': sha(path), 'output_sha256': sha(output),
            'rows': len(chosen), 'sources': dict(Counter(r['source'] for r in chosen)),
            'overlength_rows_skipped_before_quota': dict(excluded),
            'max_tokens_per_model': np.max(lengths, axis=0).tolist(),
            'median_tokens_per_model': np.median(lengths, axis=0).tolist()}
    assert not split_states[0] & split_states[1], 'Train/validation state overlap'
    report['exact_train_validation_state_overlap'] = 0
    write(HERE / 'data_manifest.json', report)
    print(json.dumps(report), flush=True)


@torch.no_grad()
def evaluate(model, data):
    was_training = model.training
    model.eval()
    # Warm up; exclude first-use kernel compilation from inference measurements.
    for ex in data[:3]:
        label_logits(model, ex)
    torch.cuda.synchronize()
    rows = []
    for index, ex in enumerate(data):
        torch.cuda.synchronize()
        started = time.perf_counter()
        z = label_logits(model, ex)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - started) * 1000
        target = torch.tensor(ex['target'], device='cuda', dtype=torch.float32)
        nll = float(-(target * z.log_softmax(-1)).sum())
        rows.append({'index': index, 'source': ex['source'], 'type': ex['type'], 'tokens': len(ex['ids']),
            'nll': nll, 'correct': int(z.argmax() == target.argmax()), 'latency_ms': ms,
            'probabilities': z.softmax(-1).cpu().tolist()})
    by_source = {}
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        by_source[source] = {'n': len(group), 'acc': np.mean([r['correct'] for r in group]),
                             'nll': np.mean([r['nll'] for r in group])}
    result = {'n': len(rows), 'acc': np.mean([r['correct'] for r in rows]),
        'nll': np.mean([r['nll'] for r in rows]),
        'latency_median_ms': np.median([r['latency_ms'] for r in rows]),
        'latency_mean_ms': np.mean([r['latency_ms'] for r in rows]),
        'latency_p95_ms': np.percentile([r['latency_ms'] for r in rows], 95), 'by_source': by_source}
    model.train(was_training)
    return result, rows


def run(index, smoke=False):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.set_per_process_memory_fraction(.88)
    spec = json.loads((HERE / 'models.json').read_text())[index]
    name = 'qwen3-4b' if index == 0 else 'qwen35-4b'
    out = ROOT / 'adapters' / 'jet-4b-comparison' / (name + ('-smoke' if smoke else ''))
    out.mkdir(parents=True, exist_ok=False)
    config = {'model': spec, 'rank': 16, 'alpha': 16, 'dropout': .05, 'lr': 1e-4,
        'microbatch': 1, 'accumulate': 4, 'epochs': 1, 'seed': SEED,
        'max_prompt': MAX_PROMPT, 'dtype': 'bfloat16', 'quantization': None,
        'gradient_checkpointing': True, 'train_sha256': sha(HERE / 'train.jsonl'),
        'validation_sha256': sha(HERE / 'validation.jsonl'), 'script_sha256': sha(__file__),
        'backend_sha256': sha(ROOT / 'src/qwen35_training.py'), 'smoke': smoke,
        'torch': torch.__version__, 'selection': 'lowest validation NLL, initial eligible'}
    write(out / 'config.json', config)
    if index == 1:
        model, tok, info = load_model(spec['repo'], spec['revision'], train=True, rank=16)
    else:
        tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
        model, info = Qwen3ForCausalLM.from_pretrained(spec['repo'], revision=spec['revision'],
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa', output_loading_info=True)
        assert not info.get('missing_keys') and not info.get('mismatched_keys'), info
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=.05,
            target_modules=TARGETS, bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    assert {p.dtype for p in model.parameters() if not p.requires_grad} == {torch.bfloat16}
    train = examples(HERE / 'train.jsonl', tok, MAX_PROMPT)
    val = examples(HERE / 'validation.jsonl', tok, MAX_PROMPT)
    params = [p for p in model.parameters() if p.requires_grad]
    config.update(trainable_parameters=sum(p.numel() for p in params), loading_info=info)
    write(out / 'config.json', config)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    if smoke:
        order = sorted(train, key=lambda x: len(x['ids']))
        train = [order[len(order)//2], order[-1]]
        val = train
        accumulation = 1
    else:
        accumulation = 4
    initial, rows = evaluate(model, val)
    write(out / 'initial.json', {'metrics': initial, 'rows': rows})
    print('INITIAL', json.dumps(initial), flush=True)
    model.save_pretrained(out / 'best')
    tok.save_pretrained(out / 'best')
    best, best_step = initial, 0
    random.Random(SEED).shuffle(train)
    model.train()
    started = time.perf_counter()
    tokens = 0
    n_steps = (len(train) + accumulation - 1) // accumulation
    for offset in range(0, len(train), accumulation):
        step = offset // accumulation + 1
        group = train[offset:offset+accumulation]
        optimizer.zero_grad(set_to_none=True)
        lr = 1e-4 * min(1., step / 10) * (.1 + .9 * .5 * (1 + np.cos(np.pi * max(0, step-10) / max(1, n_steps-10))))
        for g in optimizer.param_groups:
            g['lr'] = lr
        losses = []
        for ex in group:
            z = label_logits(model, ex)
            loss = decision_loss(z, torch.tensor(ex['target']), ex['type'] == 'score', 1.)
            assert torch.isfinite(loss), 'Non-finite loss'
            (loss / len(group)).backward()
            losses.append(loss.item())
            tokens += len(ex['ids'])
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 5 == 0 or smoke or step == n_steps:
            progress = {'step': step, 'total': n_steps, 'loss': np.mean(losses), 'lr': lr,
                'grad_norm': float(norm), 'seconds': time.perf_counter()-started,
                'tokens_per_second': tokens/(time.perf_counter()-started),
                'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9}
            print('TRAIN', json.dumps(progress), flush=True)
            with (out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(progress)+'\n')
    training_seconds = time.perf_counter() - started
    final, final_rows = evaluate(model, val)
    write(out / 'final.json', {'metrics': final, 'rows': final_rows})
    model.save_pretrained(out / 'last')
    tok.save_pretrained(out / 'last')
    if final['nll'] < best['nll']:
        best, best_step = final, n_steps
        model.save_pretrained(out / 'best')
        tok.save_pretrained(out / 'best')
    # Reload saved last adapter into the same backbone and verify logits.
    model.eval()
    with torch.no_grad():
        before = label_logits(model, val[0]).clone()
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        for parameter in params:
            parameter.zero_()
        set_peft_model_state_dict(model, load_file(str(out / 'last/adapter_model.safetensors')))
        after = label_logits(model, val[0])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    summary = {'model': spec, 'initial': initial, 'final': final, 'selected': best,
        'selected_step': best_step, 'training_seconds': training_seconds,
        'training_tokens_per_second': tokens/training_seconds,
        'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9,
        'peak_reserved_gb': torch.cuda.max_memory_reserved()/1e9,
        'adapter_reload_verified': True, 'adapter_dir': str(out), 'steps': n_steps}
    write(out / 'completed.json', summary)
    write(HERE / (name + ('-smoke' if smoke else '') + '-results.json'), summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'smoke'])
    parser.add_argument('--model', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        run(args.model, args.mode == 'smoke')
