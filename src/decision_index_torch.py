"""Native Jet label-token readout for the Decision Index PyTorch engine API."""
import hashlib
import json
import math
from pathlib import Path

import torch
from decision_index.engines import Engine
from decision_index_engine import prepare_request
from qwen35_training import load_model, label_logits


class TorchJetEngine(Engine):
    name = 'jet-torch'
    latency = 'Synchronized local PyTorch request time, including complete prompt construction; no shared-prefix cache; excludes model loading.'

    def __init__(self, model='Qwen/Qwen3.5-4B', revision=None, adapter=None,
                 calibration=None, max_tokens=8192, **options):
        super().__init__(**options)
        if revision is None:
            raise ValueError('Pin a model revision')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.88)
        if model == 'Qwen/Qwen3.5-4B' or Path(model).is_dir():
            self.model, self.tokenizer, _ = load_model(model, revision, adapter=adapter)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import snapshot_download
            path = snapshot_download(model, revision=revision, allow_patterns=[
                '*.safetensors', '*.json', '*.jinja', '*.txt'])
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                device_map={'': 'cuda'}, attn_implementation='sdpa', output_loading_info=True)
            if info.get('missing_keys') or info.get('mismatched_keys'):
                raise ValueError(f'Incomplete model load: {info}')
            self.model.eval()
            if not calibration and (Path(path) / 'calibration.json').exists():
                calibration = str(Path(path) / 'calibration.json')
        self.max_tokens = int(max_tokens)
        config = self.model.config
        text_config = getattr(config, 'text_config', config)
        if not 0 < self.max_tokens <= text_config.max_position_embeddings:
            raise ValueError('Invalid complete-prompt token limit')
        self.temperature = float(json.loads(Path(calibration).read_text())['choice']) if calibration else 1.
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid temperature')
        self.provenance = {'model': model, 'revision': revision, 'adapter': adapter,
            'max_tokens': self.max_tokens, 'choice_temperature': self.temperature,
            'policy': 'Single-order native Jet prompt, complete inputs and all options; no benchmark-specific prompts or test-label access.',
            'engine_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        for helper in ['qwen35_training.py', 'format.py', 'decision_index_engine.py']:
            self.provenance[helper+'_sha256'] = hashlib.sha256(Path(__file__).with_name(helper).read_bytes()).hexdigest()
        for key, path in [('adapter', Path(adapter)/'adapter_model.safetensors' if adapter else None),
                          ('calibration', Path(calibration) if calibration else None)]:
            if path:
                self.provenance[key+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.no_grad()
    def __call__(self, state, questions):
        prepared = prepare_request(self.tokenizer, state, questions, self.max_tokens)
        answers, raw = {}, {}
        for name, q, ids, labels in prepared:
            z = label_logits(self.model, {'ids': ids, 'labels': labels})
            p = (z.double()/self.temperature).softmax(-1).cpu().tolist()
            keys = q.keys
            answers[name] = {'type': 'choice', 'choice': keys[max(range(len(p)), key=p.__getitem__)],
                'probabilities': dict(zip(keys, p)), 'confidence': max(p)}
            raw[name] = {'prompt_tokens': len(ids), 'label_logits': z.cpu().tolist()}
        return {'model': self.name, 'answers': answers,
                'usage': {'input_tokens': sum(len(x[2]) for x in prepared)}}, raw

    def synchronize(self):
        torch.cuda.synchronize()

    def runtime(self):
        import transformers
        return {'torch': torch.__version__, 'transformers': transformers.__version__,
                'device': torch.cuda.get_device_name(), 'backend': 'PyTorch BF16 / SDPA / FLA; native convolution'}
