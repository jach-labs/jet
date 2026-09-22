"""LoRA fine-tuning of a small LM into a Jet decision model (MLX, Apple Silicon).

The loss is cross-entropy between the soft target and the model's next-token
distribution restricted to the question's label tokens, read at the last prompt
position only - exactly what inference computes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

from jet.format import Question, label_token_ids
from jet.model import DEFAULT_BASE_MODEL, encode, label_logits, pad_batch, pad_labels


class Example:
    __slots__ = ("tokens", "labels", "target")

    def __init__(self, tokens: list[int], labels: list[int], target: list[float]):
        self.tokens, self.labels, self.target = tokens, labels, target


def load_examples(path: Path, tokenizer, max_state_tokens: int, smoothing: float) -> list[Example]:
    out = []
    for line in path.open():
        ex = json.loads(line)
        q = Question.from_dict(ex["question"])
        k = len(q.keys)
        target = [(1 - smoothing) * t + smoothing / k for t in ex["target"]]
        out.append(Example(encode(tokenizer, ex["state"], q, max_state_tokens), label_token_ids(tokenizer, q), target))
    return out


def make_batches(examples: list[Example], max_batch_tokens: int, rng: random.Random | None) -> list[list[Example]]:
    """Group similar lengths together; each batch's padded size stays under max_batch_tokens."""
    order = sorted(examples, key=lambda e: len(e.tokens))
    batches, cur = [], []
    for ex in order:
        width = max(len(ex.tokens), max((len(e.tokens) for e in cur), default=0))
        if cur and width * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(ex)
    if cur:
        batches.append(cur)
    if rng:
        rng.shuffle(batches)
    return batches


def to_arrays(batch: list[Example]):
    tokens, lengths = pad_batch([e.tokens for e in batch])
    ids, mask = pad_labels([e.labels for e in batch])
    target = np.zeros(ids.shape, dtype=np.float32)
    for i, e in enumerate(batch):
        target[i, : len(e.target)] = e.target
    return tokens, lengths, ids, mask, mx.array(target)


def loss_fn(model, tokens, lengths, ids, mask, target):
    logits = label_logits(model, tokens, lengths, ids, mask)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(target * logp).sum(axis=-1).mean()


def evaluate(model, batches: list[list[Example]]) -> dict[str, float]:
    total, nll, correct = 0, 0.0, 0
    for batch in batches:
        tokens, lengths, ids, mask, target = to_arrays(batch)
        logits = label_logits(model, tokens, lengths, ids, mask)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll += float(-(target * logp).sum())
        correct += int((mx.argmax(logits, axis=-1) == mx.argmax(target, axis=-1)).sum())
        total += len(batch)
    return {"nll": nll / total, "acc": correct / total}


def save_adapter(model, out: Path, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a Jet decision model with LoRA.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("adapters/jet"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-scale", type=float, default=10.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lora-layers", type=int, default=-1, help="-1 = all layers")
    ap.add_argument("--max-batch-tokens", type=int, default=4096)
    ap.add_argument("--max-state-tokens", type=int, default=1024)
    ap.add_argument("--smoothing", type=float, default=0.02)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-limit", type=int, default=1000)
    ap.add_argument("--resume", type=Path, default=None, help="adapter dir to continue from")
    ap.add_argument("--start-step", type=int, default=0, help="step the resumed adapter was saved at (keeps the LR schedule aligned)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    model, tokenizer = load(args.base_model)
    model.freeze()
    num_layers = len(model.layers) if args.lora_layers < 0 else args.lora_layers
    lora_parameters = {"rank": args.rank, "scale": args.lora_scale, "dropout": args.dropout}
    linear_to_lora_layers(model, num_layers, lora_parameters)
    if args.resume:
        model.load_weights(str(args.resume / "adapters.safetensors"), strict=False)
        print(f"resumed from {args.resume} at step {args.start_step}")
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.2f}M")

    train = load_examples(args.train, tokenizer, args.max_state_tokens, args.smoothing)
    val = load_examples(args.val, tokenizer, args.max_state_tokens, 0.0)
    rng.shuffle(val)
    val_batches = make_batches(val[: args.val_limit], args.max_batch_tokens, None)
    steps_per_epoch = len(make_batches(train, args.max_batch_tokens, None))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    print(f"train {len(train)} examples, {steps_per_epoch} steps/epoch, {total_steps} steps; val {len(val_batches)} batches")

    warmup = max(1, min(args.warmup, total_steps // 10))
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.lr, warmup), optim.cosine_decay(args.lr, total_steps - warmup, args.lr * 0.05)],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    optimizer.state["step"] = mx.array(args.start_step, mx.uint64)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_parameters,
        "base_model": args.base_model,
    }
    model.eval()
    best = evaluate(model, val_batches)
    print(f"step 0 val nll {best['nll']:.4f} acc {best['acc']:.3f}")
    model.train()

    saved = False
    step, seen_tokens, started = args.start_step, 0, time.perf_counter()
    losses: list[float] = []
    while step < total_steps:
        for batch in make_batches(train, args.max_batch_tokens, rng):
            if step >= total_steps:
                break
            arrays = to_arrays(batch)
            loss, grads = value_and_grad(model, *arrays)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state, loss)
            step += 1
            losses.append(loss.item())
            seen_tokens += int(arrays[1].sum())
            if step % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step {step}/{total_steps} loss {np.mean(losses):.4f} lr {optimizer.learning_rate.item():.2e} "
                    f"{seen_tokens / elapsed:.0f} tok/s peak mem {mx.get_peak_memory() / 1e9:.1f}GB",
                    flush=True,
                )
                losses = []
            if step % args.eval_every == 0 or step == total_steps:
                model.eval()
                metrics = evaluate(model, val_batches)
                model.train()
                improved = metrics["nll"] < best["nll"]
                print(f"step {step} val nll {metrics['nll']:.4f} acc {metrics['acc']:.3f}{'  *saved*' if improved else ''}", flush=True)
                if improved:
                    best = metrics
                    save_adapter(model, args.out, adapter_config)
                    saved = True
    if not saved:
        print("val never improved on the base model; saving the final weights anyway")
        save_adapter(model, args.out, adapter_config)
    print(f"best val nll {best['nll']:.4f} acc {best['acc']:.3f}; adapter in {args.out}")


if __name__ == "__main__":
    main()
