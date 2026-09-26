import unittest
from train import GUARDS,eligible,family
class SelectionTests(unittest.TestCase):
 def test_replay_source_is_retention(self):
  self.assertEqual(family('retention:v3:sarcasm_irony'),'retention')
  self.assertEqual(family('focused:sarcasm'),'sarcasm')
  self.assertEqual(family('rules:conditional'),'conditional')
 def test_no_family_can_be_hidden_by_aggregate_gain(self):
  baseline={'families':{k:{'metric':.8} for k in GUARDS}}
  for family in GUARDS:
   candidate={'families':{k:{'metric':.77 if k==family else .95} for k in GUARDS}}
   self.assertFalse(eligible(candidate,baseline))
if __name__=='__main__':unittest.main()
