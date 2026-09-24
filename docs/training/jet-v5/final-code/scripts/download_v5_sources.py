"""Pin native training and exclusion-only evaluation partitions for V5."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
SPECS = [
 ('clinc/clinc_oos','155b9c710419136e17307b80d0a13e68cd46b4ec',[f'plus/{s}-00000-of-00001.parquet' for s in ['train','validation','test']]),
 ('Rowan/hellaswag','218ec52e09a7e7462a5400043bb9a69a41d06b76',[f'data/{s}-00000-of-00001.parquet' for s in ['train','validation','test']]),
 ('facebook/anli','8e4813d81f46d313dac7892e1c28076917cfcdf9',['plain_text/train_r3-00000-of-00001.parquet']+[f'plain_text/{s}_r{r}-00000-of-00001.parquet' for s in ['dev','test'] for r in [1,2,3]]),
 ('nvidia/When2Call','0582f7749df63a96fdc3070932e83e72396ace53',['train/when2call_train_pref.jsonl','test/when2call_test_mcq.jsonl'])]
def get(spec):
 repo,rev,file=spec
 p=hf_hub_download(repo,file,repo_type='dataset',revision=rev)
 print(repo,file,flush=True)
 return dict(repo=repo,revision=rev,file=file,path=p,role='train' if 'train' in file else 'exclusion_only')
if __name__=='__main__':
 with ThreadPoolExecutor(4) as pool: rows=list(pool.map(get,[(r,v,f) for r,v,files in SPECS for f in files]))
 Path('work/jet-v5/sources-complete.json').write_text(json.dumps(rows,indent=2)+'\n')
