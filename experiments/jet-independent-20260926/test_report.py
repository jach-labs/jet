import unittest
from report import passes
class GateTests(unittest.TestCase):
 def test_improvement_cannot_hide_family_regression(self):
  b={'objective':.6,'families':{'rule':{'metric':.6},'sarcasm':{'metric':.5}}}
  c={'objective':.8,'families':{'rule':{'metric':.8},'sarcasm':{'metric':.47}}}
  self.assertFalse(passes(c,b));c['families']['sarcasm']['metric']=.5;self.assertTrue(passes(c,b))
 def test_unchanged_fallback_cannot_pass(self):
  b={'objective':.6,'families':{'rule':{'metric':.6}}};self.assertFalse(passes(b,b))
if __name__=='__main__':unittest.main()
