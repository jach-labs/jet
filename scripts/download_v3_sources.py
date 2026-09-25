import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
specs=[('tasksource/esci','8113b17a5d4099e20243282c926f1bc1a08a4d13','data/train-00000-of-00011-2d36455632bef8a2.parquet'),('stanfordnlp/snli','cdb5c3d5eed6ead6e5a341c8e56e669bb666725b','plain_text/train-00000-of-00001.parquet'),('nyu-mll/glue','bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c','qnli/train-00000-of-00001.parquet'),('glaiveai/glaive-function-calling-v2','e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac','glaive-function-calling-v2.json')]+[('cardiffnlp/tweet_eval','b3a375baf0f409c77e6bc7aa35102b7b3534f8be',x+'/train-00000-of-00001.parquet') for x in ['irony','stance_abortion','stance_atheism','stance_climate','stance_feminist','stance_hillary']]
def get(s):
 repo,rev,file=s
 p=hf_hub_download(repo,file,repo_type='dataset',revision=rev)
 if file.endswith('.parquet'):
  t=pq.read_table(p); print(repo,file,t.schema,t.slice(0,1).to_pylist(),flush=True)
 else:
  d=json.load(open(p));print(repo,len(d),d[0],flush=True)
 return dict(repo=repo,revision=rev,file=file,path=p,split='train')
with ThreadPoolExecutor(5) as pool: entries=list(pool.map(get,specs))
Path('work/jet-v3/sources.json').write_text(json.dumps(entries,indent=2))
