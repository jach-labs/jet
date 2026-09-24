"""Backend-independent Jet prompt encoding and typed answers."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from format import Question, build_prompt, render_state

DEFAULT_MAX_STATE_TOKENS = 4096

def prompt_text(tokenizer, state: Any, q: Question, max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> str:
    """Tokenize a prompt, truncating the middle of an over-long state."""
    text = render_state(state)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_state_tokens:
        half = max_state_tokens // 2
        text = tokenizer.decode(ids[:half]) + "\n[...]\n" + tokenizer.decode(ids[-half:])
    return build_prompt(tokenizer, text, q)

def encode(tokenizer, state: Any, q: Question, max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS) -> list[int]:
    return tokenizer.encode(prompt_text(tokenizer, state, q, max_state_tokens), add_special_tokens=False)

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
