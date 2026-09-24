"""Check V4 isolation, public annotations and every generated executable target."""
import ast
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from data.decision_training import read, digest, statekey
from build_v4_transfer import ROOT, RAW, PROGRAMS

def main():
    paths={s:ROOT/'data'/p for s,p in dict(train='train_v4.jsonl',val='selection_v4.jsonl',calibration='calibration_v4.jsonl',test='test_v4_new.jsonl').items()}
    owners={};identities={};counts=Counter();positions=Counter()
    vast={r['new_id']:r for r in csv.DictReader((RAW/'vast/data/VAST/vast_train.csv').open())}
    z=zipfile.ZipFile(RAW/'repos/contractnli/resources/contract-nli.zip')
    c=json.loads(z.read('contract-nli/train.json'));contracts={str(d['id']):d for d in c['documents']}
    safe=dict(enumerate=enumerate,range=range,len=len,zip=zip,sum=sum,sorted=sorted,reversed=reversed,list=list)
    for split,path in paths.items():
        for r in read(path):
            key=statekey(r['state']); assert owners.setdefault(key,split)==split,('state overlap',split)
            if not r.get('source','').startswith('v4:'):continue
            p=r['provenance']; assert p['split']==split
            for key in [p['component'],p['origin']]: assert identities.setdefault(key,split)==split,key
            q=r['question'];idx=r['target'].index(1.0);gold=list(q['criteria'])[idx];chosen=list(q['criteria'].values())[idx]
            assert gold==r['gold']; assert sum(r['target'])==1.0
            family=r['family'];counts[split+':'+family]+=1;positions[(family,idx)]+=1
            if family=='code_transfer':
                ns={};exec(r['state']['code'],{'__builtins__':{},**safe},ns)
                out=ns['f'](**r['state']['arguments']);assert ast.literal_eval(chosen)==out==p['oracle']['result']
                assert len(set(q['criteria'].values()))==len(q['criteria'])
            elif family=='stance_transfer':
                source=vast[p['origin'].split(':')[-1]]
                assert r['state']['text']==source['post'] and r['state']['target']==source['new_topic']
                assert gold==['against','favor','none'][int(source['label'])]
            elif family=='contract_entailment':
                _,did,lid=p['origin'].split(':');source=contracts[did]
                assert r['state']['contract']==source['text']
                assert r['state']['hypothesis']==c['labels'][lid]['hypothesis']
                assert gold==source['annotation_sets'][0]['annotations'][lid]['choice']
    report=dict(status='passed',counts=dict(counts),gold_positions={str(k):v for k,v in positions.items()},checks=['No state or component crosses a split','All VAST and ContractNLI targets checked against native train annotations','Every generated program executed and selected answer verified'])
    out=ROOT/'docs/training/jet-next/v4-data-audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
