"""Typed inference for the Jet 4B adapter. Linux + NVIDIA CUDA, Python 3.12."""
import json
from pathlib import Path
import torch
from format import Question, label_token_ids
from inference import encode, summarize
from qwen35_training import load_model, label_logits

BASE_MODEL = 'Qwen/Qwen3.5-4B'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

class Jet4B:
    def __init__(self, adapter=None, max_tokens=8192):
        self.path = Path(adapter) if adapter else Path(__file__).resolve().parent
        if not 0 < max_tokens <= 8192:
            raise ValueError('This release supports complete prompts up to 8192 tokens.')
        self.max_tokens = max_tokens
        torch.set_num_threads(4)
        self.temperatures = json.loads((self.path / 'calibration.json').read_text())
        self.model, self.tokenizer, _ = load_model(BASE_MODEL, BASE_REVISION, adapter=str(self.path))

    @torch.no_grad()
    def decide(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError('questions must be a nonempty mapping')
        prepared = []
        for name, value in questions.items():
            q = Question.from_dict(value)
            ids = encode(self.tokenizer, state, q, 10**9)
            if len(ids) > self.max_tokens:
                raise ValueError(f'{name}: complete prompt exceeds {self.max_tokens} tokens; no truncation applied')
            prepared.append((name, q, {'ids': ids, 'labels': label_token_ids(self.tokenizer, q)}))
        answers = {}
        for name, q, example in prepared:
            z = label_logits(self.model, example)
            probabilities = (z.double() / self.temperatures[q.type]).softmax(-1).cpu().numpy()
            answers[name] = summarize(q, probabilities)
        return {'model': 'michaljach/jet-4b', 'answers': answers}

if __name__ == '__main__':
    import sys
    request = json.load(sys.stdin)
    print(json.dumps(Jet4B().decide(request['state'], request['questions']), indent=2))
