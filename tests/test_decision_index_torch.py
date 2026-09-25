"""Schema preservation and whole-request capacity checks, without model weights."""
import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from decision_index.engines import Unsupported, validate
from decision_index_torch import TorchJetEngine


class TorchEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B',
            revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)

    def engine(self, limit=8192):
        engine=object.__new__(TorchJetEngine)
        engine.tokenizer=self.tokenizer;engine.max_tokens=limit
        engine.temperature=2.;engine.model=None
        return engine

    def test_all_options_and_question_keys_preserved(self):
        questions={name:{'type':'choice','instructions':'Choose one.',
            'criteria':{f'key-{i}':f'Option {i}' for i in range(n)}} for name,n in [('wide',151),('small',2)]}
        with patch('decision_index_torch.label_logits',side_effect=lambda m,e:torch.arange(len(e['labels']),dtype=torch.float32)):
            response,raw=self.engine()('Test state',questions)
        validate(questions,response)
        self.assertEqual(list(response['answers']['wide']['probabilities']),list(questions['wide']['criteria']))
        self.assertEqual(response['answers']['wide']['choice'],'key-150')
        self.assertEqual(len(raw['wide']['label_logits']),151)

    def test_capacity_rejects_whole_request_before_inference(self):
        q={'q':{'type':'choice','instructions':'Choose.','criteria':{'a':'A','b':'B'}}}
        with patch('decision_index_torch.label_logits') as call:
            with self.assertRaises(Unsupported):self.engine(1)('Long input',q)
        call.assert_not_called()

    def test_unsupported_type_is_explicit(self):
        with self.assertRaises(Unsupported):
            self.engine()('State',{'q':{'type':'noul','instructions':'Yes?'}})

if __name__=='__main__':unittest.main()
