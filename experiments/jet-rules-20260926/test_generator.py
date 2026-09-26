import datetime as dt,itertools,json,unittest
from generate import TREES,evaluate,expression,leaves,rules,deadlines,deadline_label
class GeneratorTests(unittest.TestCase):
 def test_all_truth_tables_match_independent_python_expressions(self):
  for families in TREES.values():
   for trees in families.values():
    for tree in trees:
     for bits in itertools.product([False,True],repeat=max(leaves(tree))+1):
      self.assertEqual(evaluate(tree,bits),eval(expression(tree),{'__builtins__':{}},{'b':bits}))
 def test_structures_are_split_disjoint(self):
  owners={}
  for families in TREES.values():
   for split,trees in families.items():
    for tree in trees:
     key=json.dumps(tree);self.assertEqual(owners.setdefault(key,split),split)
 def test_counterfactual_pairs_reverse_labels(self):
  for family in TREES:
   rows=rules('train',family,20)
   for a,b in zip(rows[::2],rows[1::2]):
    self.assertEqual(a['provenance']['group'],b['provenance']['group'])
    self.assertNotEqual(a['provenance']['certificate']['label'],b['provenance']['certificate']['label'])
 def test_deadline_boundaries_and_leap_year(self):
  deadline=dt.date(2024,2,28)
  for received,label in [(dt.date(2024,2,27),0),(deadline,0),(dt.date(2024,2,29),1),(dt.date(2024,3,1),2)]:self.assertEqual(deadline_label(deadline,received,1),label)
  self.assertEqual(deadline_label(dt.date(2023,12,31),dt.date(2024,1,1),1),1)
 def test_one_hot_targets_and_dates(self):
  for split in ['train','selection','test']:
   for r in deadlines(split,90):
    c=r['provenance']['certificate'];label=deadline_label(dt.date.fromisoformat(c['deadline']),dt.date.fromisoformat(c['received']),c['grace_days'])
    self.assertEqual(sum(r['target']),1);self.assertEqual(r['target'].index(1.),label)
if __name__=='__main__':unittest.main()
