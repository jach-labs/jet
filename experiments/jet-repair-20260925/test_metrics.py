import unittest
from unittest.mock import patch
import torch
import train
class MetricsTests(unittest.TestCase):
 def test_sarcasm_positive_f1_ignores_option_order(self):
  data=[{'family':'sarcasm','target':[0,1],'semantic_labels':['not sarcastic','sarcastic']}, {'family':'sarcasm','target':[0,1],'semantic_labels':['sarcastic','not sarcastic']}]
  model=torch.nn.Linear(1,1)
  with patch.object(train,'label_logits',side_effect=[torch.tensor([0.,2.]),torch.tensor([2.,0.])]):score=train.evaluate(model,data)
  self.assertAlmostEqual(score['families']['sarcasm']['metric'],2/3)
  self.assertEqual(score['families']['sarcasm']['accuracy'],.5)
  self.assertTrue(model.training)
 def test_macro_f1_uses_semantic_classes(self):
  data=[{'family':'stance','target':[1,0],'semantic_labels':['favor','against']}, {'family':'stance','target':[1,0],'semantic_labels':['against','favor']}]
  with patch.object(train,'label_logits',side_effect=[torch.tensor([2.,0.]),torch.tensor([0.,2.])]):score=train.evaluate(torch.nn.Linear(1,1),data)
  self.assertAlmostEqual(score['families']['stance']['metric'],1/3)
 def test_regression_guard(self):
  baseline={'families':{'sarcasm':{'metric':.6},'retention':{'accuracy':.9}}}
  self.assertTrue(train.eligible(baseline,baseline))
  self.assertFalse(train.eligible({'families':{'sarcasm':{'metric':.57},'retention':{'accuracy':.95}}},baseline))
  self.assertFalse(train.eligible({'families':{'sarcasm':{'metric':.65},'retention':{'accuracy':.87}}},baseline))
if __name__=='__main__':unittest.main()
