import ast
import unittest
from collections import defaultdict
from data.decision_training import generators, choice, Components

class TrainingDataTests(unittest.TestCase):
    def test_executable_labels_and_positions(self):
        positions=defaultdict(set)
        for r in generators(150):
            q=r['question'];position=r['target'].index(1.0)
            chosen=list(q['criteria'].values())[position]
            state=r['state'];family=r['family']
            positions[family].add(position)
            if family=='boolean_rules':
                expected=eval(state['allow_if'],{'__builtins__':{}},state['facts'])
                self.assertEqual(chosen,'access is allowed' if expected else 'access is denied')
            elif family=='arithmetic':
                expected=eval(state['expression'],{'__builtins__':{},'abs':abs})
                self.assertEqual(ast.literal_eval(chosen),expected)
            else:
                ns={};exec(state['code'],{'__builtins__':{},'sum':sum,'sorted':sorted,'set':set,'len':len},ns)
                self.assertEqual(ast.literal_eval(chosen),ns['f'](state['input']))
        for family,p in positions.items():self.assertGreater(len(p),1,family)

    def test_connected_relations(self):
        c=Components();c.union('query1','product1');c.union('query2','product1')
        self.assertEqual(c.find('query1'),c.find('query2'))
        self.assertNotEqual(c.find('query1'),c.find('unrelated'))

    def test_deterministic_generation(self):
        self.assertEqual(list(generators(5)),list(generators(5)))

if __name__=='__main__':unittest.main()
