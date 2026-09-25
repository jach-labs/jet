import unittest
from unittest.mock import patch
import torch
from train import evaluate,eligible,family
class FakeModel:
 training=False
 def eval(self):pass
 def train(self,value):pass
class MetricsTest(unittest.TestCase):
 def test_retention_is_not_reclassified_by_source_name(self):
  self.assertEqual(family('retention:v3:sarcasm_irony'),'retention')
  self.assertEqual(family('focused:finance'),'finance')
 def test_semantic_labels_survive_option_order(self):
  data=[dict(family='sarcasm',semantic_labels=['sarcastic','not sarcastic'],target=[1.,0.],type='choice'),dict(family='sarcasm',semantic_labels=['not sarcastic','sarcastic'],target=[1.,0.],type='choice')]
  with patch('train.label_logits',side_effect=[torch.tensor([3.,0.]),torch.tensor([0.,3.])]):
   result=evaluate(FakeModel(),data)
  self.assertAlmostEqual(result['families']['sarcasm']['metric'],2/3)
 def test_each_family_has_regression_guard(self):
  baseline={'families':{k:{'metric':.8} for k in ['banking','finance','sarcasm','retention']}}
  for family in baseline['families']:
   candidate={'families':{k:{'metric':.9 if k!=family else .77} for k in baseline['families']}}
   self.assertFalse(eligible(candidate,baseline))
if __name__=='__main__':unittest.main()
