"""Run with PyTorch and ZeroGPU requirements installed; no cloud GPU needed."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from format import Question
from torch_model import TorchJet, MODEL_ID, REVISION


class TorchGoldenTests(unittest.TestCase):
    def test_published_golden_cases(self):
        jet = TorchJet("mps" if torch.backends.mps.is_available() else "cpu")
        cases = json.loads(Path(hf_hub_download(MODEL_ID, "golden.json", revision=REVISION)).read_text())["cases"]
        for case in cases:
            with self.subTest(case=case["name"]):
                q = Question.from_dict(case["question"])
                _, _, ids, labels = jet.prepare(case["state"], {"answer": q})[0]
                self.assertEqual(ids, case["input_ids"])
                probs = jet.probabilities(ids, labels, q.type)
                self.assertEqual(int(np.argmax(probs)), int(np.argmax(case["probabilities"])))
                # Reference logits were computed in bf16; local testing uses fp32.
                self.assertLess(float(np.max(np.abs(probs - case["probabilities"]))), 0.05)
