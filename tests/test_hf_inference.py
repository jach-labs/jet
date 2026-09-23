"""Integration checks: run with the API requirements and httpx installed.

PYTHONPATH=src:deploy/huggingface/api python -m unittest discover -s tests
Downloads the released ONNX model (about 791 MB) on first use.
"""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from huggingface_hub import hf_hub_download
from app import app
from format import Question
from inference import encode
from onnx_model import MODEL_ID, REVISION


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.golden = json.loads(Path(hf_hub_download(MODEL_ID, 'golden.json', revision=REVISION)).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_golden(self):
        errors = []
        for case in self.golden['cases']:
            with self.subTest(case=case['name']):
                q = Question.from_dict(case['question'])
                self.assertEqual(encode(app.state.jet.tokenizer, case['state'], q), case['input_ids'])
                probs, _ = app.state.jet.probabilities(case['state'], q)
                self.assertEqual(int(np.argmax(probs)), int(np.argmax(case['probabilities'])))
                error = float(np.max(np.abs(probs - case['probabilities'])))
                self.assertLess(error, 0.04)
                errors.append(error)
        print(f'Golden cases: {len(errors)}, max probability difference: {max(errors):.5f}')

    def test_api(self):
        payload = {'state':'I love this product.', 'questions': {'positive': {'type':'noul','instructions':'Is the sentiment positive?'}}}
        self.assertEqual(self.client.get('/health').status_code, 200)
        self.assertEqual(self.client.post('/v1/decide', json={'state':'x','questions':{}}).status_code, 400)
        self.assertEqual(self.client.post('/v1/decide', json={**payload,'model':'unknown'}).status_code, 400)
        with patch.dict(os.environ, {'JET_API_KEY':'test-key'}):
            self.assertEqual(self.client.post('/v1/decide',json=payload).status_code, 401)
            result = self.client.post('/v1/decide',json=payload, headers={'Authorization':'Bearer test-key'})
        self.assertEqual(result.status_code,200)
        self.assertGreater(result.json()['answers']['positive']['probability'],0.5)
        app.state.busy = True
        try:
            self.assertEqual(self.client.post('/v1/decide',json=payload).status_code,503)
        finally:
            app.state.busy = False
