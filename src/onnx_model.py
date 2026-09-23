"""CPU inference using Jet's published quantized ONNX export."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from format import Question, label_token_ids
from inference import encode, summarize

MODEL_ID = "michaljach/jet"
REVISION = "8a97cfea2df622bb03f5dc9b02567e21abd2551c"


class OnnxJet:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.temperatures = json.loads(Path(hf_hub_download(MODEL_ID, "calibration.json", revision=REVISION)).read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            hf_hub_download(MODEL_ID, "onnx/model_q8.onnx", revision=REVISION),
            sess_options=options, providers=["CPUExecutionProvider"],
        )

    def probabilities(self, state, question: Question):
        tokens = encode(self.tokenizer, state, question, max_state_tokens=2048)
        if len(tokens) > 4096:
            raise ValueError("Question and state together exceed the 4096-token demo limit")
        cache = {x.name: np.zeros((1, 8, 0, 128), dtype=np.float32)
                 for x in self.session.get_inputs() if x.name.startswith("past_key_values.")}
        # Chunk prefill to bound attention memory for long states.
        for start in range(0, len(tokens), 128):
            part = tokens[start:start + 128]
            outputs = self.session.run(None, {
                "input_ids": np.array([part], dtype=np.int64),
                "attention_mask": np.ones((1, start + len(part)), dtype=np.int64),
                "position_ids": np.arange(start, start + len(part), dtype=np.int64)[None],
                **cache,
            })
            cache = {x.name.replace("present.", "past_key_values."): value
                     for x, value in zip(self.session.get_outputs()[1:], outputs[1:])}
        z = outputs[0][0, -1, label_token_ids(self.tokenizer, question)] / self.temperatures[question.type]
        probs = np.exp(z - z.max())
        return probs / probs.sum(), len(tokens)

    def decide(self, state, questions: dict[str, Question]):
        started = time.perf_counter()
        answers, count = {}, 0
        for name, question in questions.items():
            probs, tokens = self.probabilities(state, question)
            answers[name] = summarize(question, probs)
            count += tokens
        return {"model": "jet-q8", "answers": answers, "usage": {"input_tokens": count},
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
