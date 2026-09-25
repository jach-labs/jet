"""Cross-check mirrored ESCI IDs/labels and ToolRet query overlap at pinned revisions."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from decision_index.suite.build.acquire import http_fetch
root=Path(__file__).resolve().parents[1]
out=root/'docs/training/jet-v3';out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in (root/'data/train_v3.jsonl').open()]
p=root/'work/jet-v3/esci-original-examples.parquet'
sha='4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
http_fetch(p,'https://media.githubusercontent.com/media/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_examples.parquet',sha)
products={int(r['provenance']['origin'].rsplit(':',1)[1]):r for r in rows if r.get('family')=='product_relevance'}
original={r['example_id']:r for r in pq.read_table(p).to_pylist() if r['example_id'] in products}
assert original.keys()==products.keys()
mapping=dict(E='Exact',S='Substitute',C='Complement',I='Irrelevant')
for key,r in original.items():
    assert r['split']=='train'
    assert mapping[r['esci_label']]==products[key]['gold']
    assert r['query']==products[key]['state']['search_query']
(out/'esci-original-split-audit.json').write_text(json.dumps(dict(verified_rows=len(products),official_split='train',labels_and_queries_match=True,official_examples_sha256=sha),indent=2))
repo='mangopy/ToolRet-Queries';rev='b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445'
p=Path(snapshot_download(repo,repo_type='dataset',revision=rev,allow_patterns=['*/*.parquet']))
normalize=lambda x:' '.join(x.casefold().split())
queries={normalize(r['query']):r['id'] for f in p.rglob('*.parquet') for r in pq.read_table(f).to_pylist()}
overlap=[]
for r in rows:
    if r.get('family') not in ('tool_selection','tool_relevance'):continue
    q=r['state'] if isinstance(r['state'],str) else r['state']['request']
    if normalize(q) in queries:overlap.append({'origin':r['provenance']['origin'],'benchmark_id':queries[normalize(q)]})
(out/'toolret-overlap-audit.json').write_text(json.dumps(dict(repository=repo,revision=rev,queries=len(queries),overlap=overlap),indent=2))
assert not overlap, 'Do not train: tool queries overlap the evaluation corpus.'
print('Official ESCI training IDs/labels match; zero ToolRet query overlap.')
