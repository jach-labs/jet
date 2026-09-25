import collections,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
FULL=[25,32,30,44,1,3,10,4,5,12,28,29,42,11,20,21,23,39]
REUSED=[40,41,43,50,37];SAMPLED=[2,36]
def get(url):return urllib.request.urlopen(url).read()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 revision=json.loads(get('https://huggingface.co/api/spaces/multimodalart/jev-decision-index'))['sha']
 url=f'https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/{revision}/data/index-v0.1.json'
 blob=get(url);(HERE/'reference-index.json').write_bytes(blob);index=json.loads(blob)
 kev=next(m for m in index['models'] if m['engine']=='kev-8b-raised')
 (HERE/'kev-reference.json').write_text(json.dumps(kev,indent=2)+'\n')
 path=ROOT/'experiments/jet-4b-full-index/expanded.jsonl'
 groups=collections.defaultdict(set);available=collections.Counter()
 for line in path.open():
  r=json.loads(line);e=r['_evaluation'];bid=int(e['catalog_id']);available[bid]+=1
  if bid in SAMPLED:groups[(bid,e['track'])].add(e['group_id'])
 selected=set()
 for (bid,track),ids in groups.items():
  # Round-robin over tracks below, selected by a fixed hash without consulting labels.
  groups[(bid,track)]=sorted(ids,key=lambda gid:hashlib.sha256(f'250925:{bid}:{track}:{gid}'.encode()).hexdigest())
 for bid in SAMPLED:
  tracks=sorted(t for b,t in groups if b==bid);n=0;i=0
  while n<100:
   progress=False
   for track in tracks:
    ids=groups[(bid,track)]
    if i<len(ids) and n<100:selected.add((bid,track,ids[i]));n+=1;progress=True
   if not progress:break
   i+=1
 rows=collections.defaultdict(list)
 for line in path.open():
  r=json.loads(line);e=r['_evaluation'];bid=int(e['catalog_id'])
  if bid in FULL+REUSED or (bid,e['track'],e['group_id']) in selected:rows[bid].append(line)
 out=HERE/'rows.jsonl'
 with out.open('w') as f:
  for bid in REUSED+FULL+SAMPLED:f.writelines(rows[bid])
 payloads={}
 for bid in REUSED:
  for line in rows[bid]:
   e=json.loads(line)['_evaluation'];payloads[e['run_id']]=e['payload_sha256']
 old=ROOT/'artifacts/decision-index/runs/jet-targeted-eval-20260925/candidate/results.jsonl'
 run=ROOT/'artifacts/decision-index/runs'/HERE.name;run.mkdir(parents=True,exist_ok=True)
 assert not (run/'results.jsonl').exists(),'Existing run: do not overwrite'
 oldprotocol=json.loads((ROOT/'experiments/jet-targeted-eval-20260925/protocol.json').read_text())
 adapter=ROOT/'adapters/jet-targeted-20260924/best/adapter_model.safetensors'
 assert sha(adapter)==oldprotocol['candidate_sha256']
 count=0
 with (run/'results.jsonl').open('w') as f:
  for line in old.open():
   r=json.loads(line)
   if r['run_id'] in payloads:
    assert r['payload_sha256']==payloads.pop(r['run_id']);assert r['status']=='ok';f.write(line);count+=1
 assert not payloads
 audit={'reference_url':url,'space_revision':revision,'reference_sha256':hashlib.sha256(blob).hexdigest(),'reference_generated_utc':index['generated_utc'],'reference_engine':kev['engine'],'reference_scores':kev['scores'],'source_rows_sha256':sha(path),'rows_sha256':sha(out),'candidate_step':2000,'candidate_adapter_sha256':sha(adapter),'base_revision':'e5b8f610ddb92ffaba596ae452bed32a9fef49ca','reused_predictions':count,'total_requests':sum(map(len,rows.values())),'new_requests':sum(map(len,rows.values()))-count,'datasets':[],'limitations':['Kev figures are archived published scores, not a new local Kev inference run.','Matching counts and metrics do not establish byte-identical cases; the official frozen corpus is unavailable.','ToolRet and BRIGHT are 100 complete-query diagnostics, selected without outcomes; never treat their deltas as a leaderboard comparison.','No subset average or official overall Decision Index is produced.','Benchmarks have informed prior training choices; this is development evaluation.']}
 for bid in REUSED+FULL+SAMPLED:
  ref=kev['results'].get(str(bid),{});bm=index['benchmarks'][str(bid)]
  audit['datasets'].append({'id':bid,'dataset':bm['dataset'],'scope':'sampled complete query groups' if bid in SAMPLED else 'full available reconstruction','requests':len(rows[bid]),'available':available[bid],'reference_requests':ref.get('requests'),'reference_metric':ref.get('metric'),'reference_score':ref.get('score'),'reused':bid in REUSED})
 (HERE/'protocol.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({k:audit[k] for k in ['total_requests','new_requests','reused_predictions','datasets']},indent=2))
if __name__=='__main__':main()
