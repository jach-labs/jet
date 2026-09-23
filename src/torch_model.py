"""Jet inference for PyTorch, including Hugging Face ZeroGPU."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from format import Question, label_token_ids
from inference import encode, summarize

MODEL_ID = "michaljach/jet"
REVISION = "8a97cfea2df622bb03f5dc9b02567e21abd2551c"


class TorchJet:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.temperatures = json.loads(Path(hf_hub_download(
            MODEL_ID, "calibration.json", revision=REVISION,
        )).read_text())
        # ZeroGPU expects the model to move to CUDA at module initialization.
        # A real GPU is allocated only while a @spaces.GPU function is running.
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).eval().to(device)

    def prepare(self, state, questions: dict[str, Question]):
        prepared = []
        for name, q in questions.items():
            ids = encode(self.tokenizer, state, q, max_state_tokens=4096)
            if len(ids) > 6144:
                raise ValueError("State and question together exceed 6144 tokens")
            prepared.append((name, q, ids, label_token_ids(self.tokenizer, q)))
        return prepared

    @torch.inference_mode()
    def probabilities(self, ids, labels, question_type):
        tokens = torch.tensor([ids], device=self.device)
        # Apply the vocabulary head only to the last position: no generation,
        # and no sequence-length × vocabulary-size intermediate allocation.
        hidden = self.model.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1, :]
        logits = self.model.lm_head(hidden)[0, labels].float()
        return torch.softmax(logits / self.temperatures[question_type], dim=-1).cpu().numpy()

    def decide_prepared(self, prepared):
        started = time.perf_counter()
        answers, tokens = {}, 0
        for name, q, ids, labels in prepared:
            answers[name] = summarize(q, self.probabilities(ids, labels, q.type))
            tokens += len(ids)
        return {"model": "jet", "answers": answers, "usage": {"input_tokens": tokens},
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
