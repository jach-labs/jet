import unittest
from decision_index_ensemble import combine, TwoOrderJetEngine
from unittest.mock import patch
import copy
class EnsembleTests(unittest.TestCase):
 def test_alignment_and_confidence(self):
  a={'answers':{'q':{'probabilities':{' X':.9,'x':.1}}},'usage':{'input_tokens':5}}
  b={'answers':{'q':{'probabilities':{'x':.7,' X':.3}}},'usage':{'input_tokens':6}}
  r=combine(a,b)
  self.assertAlmostEqual(r['answers']['q']['probabilities'][' X'],.6)
  self.assertEqual(r['answers']['q']['choice'],' X')
  self.assertAlmostEqual(r['answers']['q']['confidence'],.6)
  self.assertEqual(r['usage']['input_tokens'],11)
  self.assertEqual(a['answers']['q']['probabilities'][' X'],.9)
 def test_reject_missing_option(self):
  a={'answers':{'q':{'probabilities':{'a':1.}}},'usage':{'input_tokens':1}}
  b={'answers':{'q':{'probabilities':{'b':1.}}},'usage':{'input_tokens':1}}
  with self.assertRaises(ValueError):combine(a,b)
 def test_complete_options_and_input_preserved(self):
  engine=TwoOrderJetEngine.__new__(TwoOrderJetEngine)
  questions={'q':{'type':'choice','instructions':'Choose','criteria':{' A':'first','a':'second','Z':'third'}}}
  original=copy.deepcopy(questions);orders=[]
  def forward(self,state,qs):
   keys=list(qs['q']['criteria']);orders.append(keys)
   return {'answers':{'q':{'probabilities':dict(zip(keys,[.5,.3,.2]))}},'usage':{'input_tokens':9}},{}
  with patch('decision_index_engine.JetEngine.__call__',forward):
   response,_=engine({'nested':['unchanged']},questions)
  self.assertEqual(orders,[[' A','a','Z'],['Z','a',' A']])
  self.assertEqual(questions,original)
  self.assertEqual(set(response['answers']['q']['probabilities']),set(original['q']['criteria']))
if __name__=='__main__':unittest.main()
