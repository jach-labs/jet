"""Typed inference for the merged Jet model. Linux + NVIDIA CUDA, Python 3.12."""
import json
from pathlib import Path
import torch
from format import Question, label_token_ids
from inference import encode, summarize
from runtime import load_model, label_logits


class Jet:
    def __init__(self, model_path=None, max_tokens=8192):
        self.path = Path(model_path) if model_path else Path(__file__).resolve().parent
        if not 0 < max_tokens <= 8192:
            raise ValueError('This release supports complete prompts up to 8192 tokens.')
        self.max_tokens = max_tokens
        torch.set_num_threads(4)
        self.temperatures = json.loads((self.path / 'calibration.json').read_text())
        self.model, self.tokenizer = load_model(str(self.path))

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
        return {'model': 'michaljach/jet', 'answers': answers}

if __name__ == '__main__':
    import sys
    request = json.load(sys.stdin)
    print(json.dumps(Jet().decide(request['state'], request['questions']), indent=2))
