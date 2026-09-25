"""Jet's native MLX readout through the official Decision Index engine API.

Install with ``uv sync --extra benchmark``. No benchmark identity or gold labels
are accepted by this adapter. The full native prompt is preserved, or rejected.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from decision_index.engines import Engine, Unsupported
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from format import MAX_CHOICE_OPTIONS, Question, build_prompt, label_token_ids

MODEL_ID = "michaljach/jet"
REVISION = "8a97cfea2df622bb03f5dc9b02567e21abd2551c"
DEFAULT_MAX_TOKENS = 8192


def as_text(value):
    """Mechanical JSON rendering for structured instructions/descriptions."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prepare_request(tokenizer, state, questions, max_tokens):
    prepared = []
    for name, original in questions.items():
        if original["type"] != "choice":
            raise Unsupported(f"Only suite choice questions supported: {original['type']}")
        criteria = original["criteria"]
        if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise Unsupported(f"Option count {len(criteria)} outside 2..{MAX_CHOICE_OPTIONS}")
        q = Question.from_dict({
            "type": "choice", "instructions": as_text(original["instructions"]),
            "criteria": {key: as_text(value) for key, value in criteria.items()},
        })
        # Do not call inference.encode: its application default truncates states.
        ids = tokenizer.encode(build_prompt(tokenizer, state, q), add_special_tokens=False)
        if len(ids) > max_tokens:
            raise Unsupported(f"Full prompt has {len(ids)} tokens; declared limit {max_tokens}")
        prepared.append((name, q, ids, label_token_ids(tokenizer, q)))
    return prepared


def load_assets(model=MODEL_ID, revision=REVISION):
    path = Path(snapshot_download(model, revision=revision, allow_patterns=[
        "config.json", "calibration.json", "tokenizer*", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "chat_template*",
    ]))
    return path, AutoTokenizer.from_pretrained(path), json.loads((path / "config.json").read_text())


class JetEngine(Engine):
    name = "jet"
    latency = "Synchronized local MLX request wall time, including full prompt construction and shared-prefix inference; excludes model loading."

    def __init__(self, model=MODEL_ID, revision=REVISION, max_tokens=DEFAULT_MAX_TOKENS,
                 max_cached_tokens=32768, adapter=None, **options):
        super().__init__(**options)
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        mx.set_cache_limit(1_000_000_000)
        path, self.tokenizer, config = load_assets(model, revision)
        self.max_tokens = int(max_tokens)
        self.max_cached_tokens = int(max_cached_tokens)
        if not 1 <= self.max_tokens <= config["max_position_embeddings"]:
            raise ValueError("max_tokens must be positive and within the checkpoint context window")
        if self.max_cached_tokens < 1:
            raise ValueError("max_cached_tokens must be positive")
        calibration = Path(adapter) / "calibration.json" if adapter else path / "calibration.json"
        self.temperature = float(json.loads(calibration.read_text())["choice"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid published choice temperature")
        # Pin weights to the SAME resolved snapshot as the tokenizer/config.
        snapshot_download(model, revision=path.name, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
        self.model, _ = load(str(path), adapter_path=adapter)
        self.model.eval()
        self.provenance = {
            "model": model, "revision": path.name, "backend": "mlx",
            "max_tokens": self.max_tokens, "checkpoint_context": config["max_position_embeddings"],
            "max_cached_tokens": self.max_cached_tokens, "choice_temperature": self.temperature,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "format_sha256": hashlib.sha256(Path(__file__).with_name("format.py").read_bytes()).hexdigest(),
            "policy": "Native Jet prompt and label-token readout; complete input or unsupported; no truncation, option pruning, or benchmark-specific prompts.",
        }
        if adapter:
            self.provenance["lora_adapter"] = str(Path(adapter).resolve())
            self.provenance["lora_weights_sha256"] = hashlib.sha256(
                (Path(adapter) / "adapters.safetensors").read_bytes()).hexdigest()
            self.provenance["calibration_sha256"] = hashlib.sha256(calibration.read_bytes()).hexdigest()

    def __call__(self, state, questions):
        from model import shared_prefix_label_logits

        prepared = prepare_request(self.tokenizer, state, questions, self.max_tokens)
        if not prepared:
            raise ValueError("At least one question is required")
        logits = shared_prefix_label_logits(
            self.model, [p[2] for p in prepared], [p[3] for p in prepared],
            max_cached_tokens=self.max_cached_tokens,
        )
        answers, raw = {}, {}
        for i, (name, q, ids, labels) in enumerate(prepared):
            z = logits[i, :len(labels)].astype(np.float64) / self.temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            if not np.isfinite(p).all():
                raise ValueError("Non-finite model probabilities")
            keys = q.keys
            answers[name] = {
                "type": "choice", "choice": keys[int(np.argmax(p))],
                "probabilities": dict(zip(keys, map(float, p))),
                "confidence": float(p.max()),
            }
            raw[name] = {"label_logits": logits[i, :len(labels)].tolist(), "prompt_tokens": len(ids)}
        return {"model": "jet", "answers": answers,
                "usage": {"input_tokens": sum(len(p[2]) for p in prepared)}}, raw

    def synchronize(self):
        self.mx.synchronize()

    def runtime(self):
        import platform
        from importlib.metadata import version
        return {"platform": platform.platform(), "mlx": version("mlx"),
                "mlx_lm": version("mlx-lm"), "device": str(self.mx.default_device())}
