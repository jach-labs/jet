"""Jet inference: one forward pass per batch of questions, no generation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from format import Question, build_prompt, label_token_ids, render_state

DEFAULT_BASE_MODEL = "mlx-community/Qwen3-0.6B-bf16"
DEFAULT_MAX_STATE_TOKENS = 4096
NEG_INF = -1e9


def label_logits(model, tokens: mx.array, lengths: mx.array, label_ids: mx.array, label_mask: mx.array) -> mx.array:
    """Next-token logits after each prompt, restricted to its label tokens.

    tokens: (B, L) right-padded prompts; lengths: (B,) true lengths;
    label_ids / label_mask: (B, K) label token ids padded to the widest question.
    Returns (B, K) logits with padded labels set to -inf.
    """
    hidden = model.model(tokens)
    last = hidden[mx.arange(tokens.shape[0]), lengths - 1][:, None, :]
    logits = head_logits(model, last)[:, 0, :]
    picked = mx.take_along_axis(logits, label_ids, axis=-1).astype(mx.float32)
    return mx.where(label_mask, picked, NEG_INF)


def head_logits(model, hidden: mx.array) -> mx.array:
    if getattr(model.args, "tie_word_embeddings", False):
        return model.model.embed_tokens.as_linear(hidden)
    return model.lm_head(hidden)


def shared_prefix_label_logits(
    model, seqs: list[list[int]], label_lists: list[list[int]], max_cached_tokens: int = 32_768
) -> np.ndarray:
    """Like label_logits, but encodes the prompts' common prefix (system + state) once.

    The prefix KV cache is replicated across a chunk of question suffixes, so each
    question only pays for its own few dozen tokens. Chunks are sized so that
    replicated prefix tokens stay under max_cached_tokens.
    """
    common = min(len(s) for s in seqs) - 1  # every suffix keeps at least one token
    for i in range(common):
        if any(s[i] != seqs[0][i] for s in seqs):
            common = i
            break
    prefix_cache = make_prompt_cache(model)
    if common:
        model.model(mx.array(seqs[0][:common])[None], cache=prefix_cache)
    prefix_state = [c.state for c in prefix_cache] if common else None
    if prefix_state:
        mx.eval(prefix_state)

    out = np.full((len(seqs), max(map(len, label_lists))), NEG_INF, dtype=np.float32)
    chunk = max(1, max_cached_tokens // max(common, 1))
    for start in range(0, len(seqs), chunk):
        suffixes = [s[common:] for s in seqs[start : start + chunk]]
        b = len(suffixes)
        cache = make_prompt_cache(model)
        if prefix_state:
            for c, (k, v) in zip(cache, prefix_state):
                c.state = (mx.repeat(k, b, axis=0), mx.repeat(v, b, axis=0))
        tokens, lengths = pad_batch(suffixes)
        hidden = model.model(tokens, cache=cache)
        last = hidden[mx.arange(b), lengths - 1][:, None, :]
        logits = head_logits(model, last)[:, 0, :]
        ids, mask = pad_labels(label_lists[start : start + chunk])
        picked = mx.where(mask, mx.take_along_axis(logits, ids, axis=-1).astype(mx.float32), NEG_INF)
        out[start : start + b, : picked.shape[1]] = np.array(picked)
    return out


def pad_batch(seqs: list[list[int]], pad: int = 0) -> tuple[mx.array, mx.array]:
    width = max(len(s) for s in seqs)
    arr = np.full((len(seqs), width), pad, dtype=np.int32)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return mx.array(arr), mx.array([len(s) for s in seqs], dtype=mx.int32)


def pad_labels(label_lists: list[list[int]]) -> tuple[mx.array, mx.array]:
    width = max(len(ls) for ls in label_lists)
    ids = np.zeros((len(label_lists), width), dtype=np.int32)
    mask = np.zeros((len(label_lists), width), dtype=bool)
    for i, ls in enumerate(label_lists):
        ids[i, : len(ls)] = ls
        mask[i, : len(ls)] = True
    return mx.array(ids), mx.array(mask)


def encode(tokenizer, state: Any, q: Question, max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> list[int]:
    """Tokenize a prompt, truncating the middle of an over-long state."""
    text = render_state(state)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_state_tokens:
        half = max_state_tokens // 2
        text = tokenizer.decode(ids[:half]) + "\n[...]\n" + tokenizer.decode(ids[-half:])
    return tokenizer.encode(build_prompt(tokenizer, text, q), add_special_tokens=False)


def summarize(q: Question, probs: np.ndarray) -> dict[str, Any]:
    """Turn a label distribution into the typed answer for the question."""
    probs = probs.astype(np.float64)
    k = len(probs)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    confidence = round(1.0 - entropy / math.log(k), 4)
    if q.type == "choice":
        keys = q.keys
        return {
            "type": "choice",
            "choice": keys[int(np.argmax(probs))],
            "probabilities": {key: round(float(p), 4) for key, p in zip(keys, probs)},
            "confidence": confidence,
        }
    if q.type == "score":
        return {
            "type": "score",
            "score": round(float(np.dot(np.arange(k), probs)), 4),
            "level": q.criteria[int(np.argmax(probs))],
            "probabilities": [round(float(p), 4) for p in probs],
            "confidence": confidence,
        }
    return {"type": "noul", "probability": round(float(probs[1]), 4), "confidence": confidence}


class Jet:
    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str | None = None,
        max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS,
    ):
        self.model, self.tokenizer = load(base_model, adapter_path=adapter_path)
        self.model.eval()
        self.max_state_tokens = max_state_tokens
        self.temperatures = {"choice": 1.0, "score": 1.0, "noul": 1.0}
        # Calibration lives next to the adapter, or inside a fused model dir.
        for directory in (adapter_path, base_model):
            if directory and (calib := Path(directory) / "calibration.json").exists():
                self.temperatures.update(json.loads(calib.read_text()))
                break

    def raw_logits(self, items: list[tuple[Any, Question]], shared_state: bool = False) -> tuple[list[np.ndarray], int]:
        """Uncalibrated label logits for (state, question) pairs, plus the number of
        input tokens. With shared_state, the common state prefix is encoded once."""
        seqs = [encode(self.tokenizer, s, q, self.max_state_tokens) for s, q in items]
        label_lists = [label_token_ids(self.tokenizer, q) for _, q in items]
        if shared_state and len(items) > 1:
            out = shared_prefix_label_logits(self.model, seqs, label_lists)
        else:
            tokens, lengths = pad_batch(seqs)
            ids, mask = pad_labels(label_lists)
            out = np.array(label_logits(self.model, tokens, lengths, ids, mask))
        return [out[i, : len(q.keys)] for i, (_, q) in enumerate(items)], sum(map(len, seqs))

    def probabilities(self, items: list[tuple[Any, Question]], shared_state: bool = False) -> tuple[list[np.ndarray], int]:
        logits, n_tokens = self.raw_logits(items, shared_state)
        result = []
        for (_, q), z in zip(items, logits):
            z = z / self.temperatures[q.type]
            z = np.exp(z - z.max())
            result.append(z / z.sum())
        return result, n_tokens

    def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        """Answer every question about one state together (shared batch)."""
        started = time.perf_counter()
        items = [(state, q) for q in questions.values()]
        probs, input_tokens = self.probabilities(items, shared_state=True)
        return {
            "answers": {name: summarize(q, p) for (name, q), p in zip(questions.items(), probs)},
            "usage": {"input_tokens": input_tokens},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
