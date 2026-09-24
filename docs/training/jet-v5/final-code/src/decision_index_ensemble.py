"""Two-order Jet readout; averages probabilities by original option key.

Applies uniformly to every choice question. Preserves all inputs and refuses
requests if either complete prompt exceeds capacity. No benchmark-specific data.
"""
import hashlib
from pathlib import Path
import numpy as np
from decision_index_engine import JetEngine


def combine(first, second):
    answers={}
    if set(first['answers'])!=set(second['answers']):raise ValueError('Question mismatch')
    for name,a in first['answers'].items():
        b=second['answers'][name]
        keys=list(a['probabilities'])
        if set(keys)!=set(b['probabilities']):raise ValueError('Option mismatch')
        p=np.array([(a['probabilities'][k]+b['probabilities'][k])/2 for k in keys],dtype=float)
        if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:raise ValueError('Invalid probabilities')
        p/=p.sum()
        answers[name]={'type':'choice','choice':keys[int(p.argmax())],
                       'probabilities':dict(zip(keys,map(float,p))),'confidence':float(p.max())}
    return {'model':'jet-two-order','answers':answers,'usage':{'input_tokens':first['usage']['input_tokens']+second['usage']['input_tokens']}}


class TwoOrderJetEngine(JetEngine):
    name='jet-two-order'
    def __init__(self,**options):
        super().__init__(**options)
        self.provenance.update(choice_readout='Mean of original and reversed option-order probabilities, aligned by exact original keys.',
                               ensemble_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def __call__(self,state,questions):
        original,raw_original=super().__call__(state,questions)
        reversed_questions={name:dict(q,criteria=dict(reversed(list(q['criteria'].items())))) for name,q in questions.items()}
        reverse,raw_reverse=super().__call__(state,reversed_questions)
        return combine(original,reverse),{'original':raw_original,'reversed':raw_reverse,'single_order_answers':original['answers']}
