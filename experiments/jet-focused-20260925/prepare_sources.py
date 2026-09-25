"""Download immutable training sources. Raw data stays in ignored work/."""
import hashlib,json,urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'work'/Path(__file__).resolve().parent.name;OUT.mkdir(parents=True,exist_ok=True)
sources=[]
repo='mteb/banking77';revision='18072d2685ea682290f7b8924d94c62acc19c0b2';file='data/train-00000-of-00001.parquet';name='banking-train.parquet'
raw=Path(hf_hub_download(repo,file,repo_type='dataset',revision=revision)).read_bytes();(OUT/name).write_bytes(raw)
sources.append(dict(repo=repo,revision=revision,file=file,local=name,sha256=hashlib.sha256(raw).hexdigest()))
revision='eba43610fe3cb57a3e5773cd6c037da1f8992ad5'
for file in ['SEntFiN.csv','LICENSE','README.md']:
 raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/pyRis/SEntFiN/{revision}/{file}').read();(OUT/file).write_bytes(raw)
 sources.append(dict(repo='https://github.com/pyRis/SEntFiN',revision=revision,file=file,local=file,sha256=hashlib.sha256(raw).hexdigest()))
(OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
