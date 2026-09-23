"""Fast adapter contract tests; no model weights or accelerator required."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from decision_index.engines import Unsupported, validate

from decision_index_engine import JetEngine, prepare_request


class Tokenizer:
    def encode(self, text, **kwargs):
        # Provide a large single-token label vocabulary, character tokens otherwise.
        return [sum(ord(c) * 256**i for i, c in enumerate(text))] if len(text) <= 2 else list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)


def question(criteria=None):
    return {"type": "choice", "instructions": "Choose the matching value.",
            "criteria": criteria or {"x": "one", "y": "two"}}


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_long_state_is_preserved_and_boundary_is_exact(self):
        state = "first " + "a" * 5000 + " MIDDLE_SENTINEL " + "b" * 5000 + " last"
        q = {"answer": question()}
        p = prepare_request(self.tokenizer, state, q, 20000)
        text = bytes(p[0][2]).decode()
        self.assertIn(state, text)
        length = len(p[0][2])
        self.assertEqual(prepare_request(self.tokenizer, state, q, length), p)
        with self.assertRaises(Unsupported):
            prepare_request(self.tokenizer, state, q, length - 1)

    def test_full_option_set_and_case_sensitive_keys(self):
        criteria = {"BB": "genotype", "Bb": "genotype", " pantry ": "same text", "pantry": "same text"}
        p = prepare_request(self.tokenizer, {}, {"a": question(criteria)}, 10000)
        self.assertEqual(p[0][1].criteria, criteria)
        self.assertEqual(len(p[0][3]), 4)
        wide = {str(i): f"description {i}" for i in range(255)}
        p = prepare_request(self.tokenizer, "state", {"a": question(wide)}, 100000)
        self.assertEqual(p[0][1].criteria, wide)
        self.assertEqual(len(p[0][3]), 255)
        with self.assertRaises(Unsupported):
            prepare_request(self.tokenizer, "state", {"a": question(wide | {"extra": "extra"})}, 100000)

    def test_structured_fields_are_rendered_without_mutating_input(self):
        q = question()
        q["instructions"] = {"rule": "A or B", "priority": [1, 2]}
        q["criteria"]["x"] = {"description": "object"}
        p = prepare_request(self.tokenizer, {"state": [1, 2]}, {"a": q}, 10000)
        self.assertIsInstance(q["instructions"], dict)
        self.assertIn('"rule": "A or B"', p[0][1].instructions)
        self.assertIn('"description": "object"', p[0][1].criteria["x"])

    def test_probabilities_validate_and_are_not_rounded(self):
        engine = JetEngine.__new__(JetEngine)
        engine.tokenizer = self.tokenizer
        engine.max_tokens = 100000
        engine.max_cached_tokens = 32768
        engine.temperature = 1.3
        engine.model = object()
        q = {"a": question(), "b": question({str(i): str(i) for i in range(255)})}
        z = np.zeros((2, 255), dtype=np.float32)
        z[0, 0] = 0.123456
        with patch("model.shared_prefix_label_logits", return_value=z):
            response, _ = engine("state", q)
        validate(q, response)
        for answer in response["answers"].values():
            self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
            self.assertEqual(answer["confidence"], max(answer["probabilities"].values()))
        self.assertEqual(response["answers"]["b"]["probabilities"]["0"], 1 / 255)

    def test_one_unsupported_question_prevents_partial_inference(self):
        engine = JetEngine.__new__(JetEngine)
        engine.tokenizer = self.tokenizer
        engine.max_tokens = 1000
        engine.max_cached_tokens = 1000
        q = question()
        q["instructions"] = "long " * 1000
        with patch("model.shared_prefix_label_logits") as forward:
            with self.assertRaises(Unsupported):
                engine("state", {"short": question(), "long": q})
            forward.assert_not_called()

    def test_diagnostic_report_never_emits_a_headline_index(self):
        from bench_decision_index import report, write_rows
        import json
        rows = [{"state": "state", "questions": {"answer": question()},
                 "expected": {"answer": "x"},
                 "_evaluation": {"run_id": "44:test:1", "catalog_id": 44,
                                 "dataset": "CLadder", "group_id": "1", "track": "CLadder"}}]
        results = [{"run_id": "44:test:1", "catalog_id": 44, "status": "ok",
                    "total_wall_ms": 1.0,
                    "response": {"answers": {"answer": {"choice": "x", "probabilities": {"x": 0.8, "y": 0.2}}}}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_rows(path / "rows.jsonl", rows)
            write_rows(path / "results.jsonl", results)
            report(path / "rows.jsonl", path / "results.jsonl", path / "report.json")
            data = json.loads((path / "report.json").read_text())
        self.assertNotIn("decision_index", data)
        self.assertEqual(data["summary"]["counts"], {"ok": 1})


@unittest.skipUnless(os.environ.get("JET_RUN_MODEL_TESTS") == "1", "Opt in to model download/inference")
class PublishedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = JetEngine()

    def test_published_choice_golden_cases(self):
        import json
        from huggingface_hub import hf_hub_download
        from decision_index_engine import MODEL_ID, REVISION
        cases = json.loads(Path(hf_hub_download(MODEL_ID, "golden.json", revision=REVISION)).read_text())["cases"]
        for case in cases:
            if case["question"]["type"] != "choice":
                continue
            with self.subTest(case=case["name"]):
                questions = {"answer": case["question"]}
                prepared = prepare_request(self.engine.tokenizer, case["state"], questions, self.engine.max_tokens)
                self.assertEqual(prepared[0][2], case["input_ids"])
                response, _ = self.engine(case["state"], questions)
                validate(questions, response)
                p = list(response["answers"]["answer"]["probabilities"].values())
                self.assertEqual(int(np.argmax(p)), int(np.argmax(case["probabilities"])))
                self.assertLess(float(np.max(np.abs(np.array(p) - case["probabilities"]))), 0.01)

    def test_shared_prefix_matches_independent_native_readout(self):
        from model import label_logits, pad_batch, pad_labels
        import mlx.core as mx
        state = "A customer requests a refund after receiving a damaged parcel. " * 12
        questions = {
            "topic": {"type": "choice", "instructions": "What is the topic?", "criteria": {"refund": "refund", "spam": "spam"}},
            "action": {"type": "choice", "instructions": "What should the support agent do next? Consider the customer's request and the damage to the parcel.",
                       "criteria": {"investigate": "Investigate damage and refund eligibility", "ignore": "Ignore the request", "promote": "Send an advertisement"}},
        }
        _, raw = self.engine(state, questions)
        for name, q, ids, labels in prepare_request(self.engine.tokenizer, state, questions, self.engine.max_tokens):
            tokens, lengths = pad_batch([ids])
            label_ids, mask = pad_labels([labels])
            z = np.array(label_logits(self.engine.model, tokens, lengths, label_ids, mask))[0]
            np.testing.assert_allclose(raw[name]["label_logits"], z, atol=0.25, rtol=0.02)
        mx.synchronize()


if __name__ == "__main__":
    unittest.main()
