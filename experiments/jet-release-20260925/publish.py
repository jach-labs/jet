import os
os.environ['HF_HUB_DISABLE_XET']='1'
import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,ModelCard,CommitOperationAdd,CommitOperationDelete
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;P=ROOT/'releases/jet-v6.1';api=HfApi();repo='michaljach/jet'
manifest=json.loads((P/'release-manifest.json').read_text());assert manifest['validation']['passed']
for name,expected in manifest['files'].items():
 with (P/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
ModelCard.load(P/'README.md').validate()
head=api.model_info(repo).sha
assert head=='9e90b3f095769e42432fb6f8c807fa4aa0c73dde',f'Remote changed: {head}'
parent=json.loads((ROOT/'releases/jet-v6/release-manifest.json').read_text())['files']
for r in api.list_repo_tree(repo,revision=head,recursive=True):
 if r.path.endswith('.safetensors'):assert r.lfs.sha256==parent[r.path],r.path
api.create_tag(repo,tag='v6.0.0',revision=head,tag_message='Previous Jet full merged model before v6.1 continuation',exist_ok=True)
files=[p for p in P.iterdir() if p.is_file()];names={p.name for p in files};existing=api.list_repo_files(repo,revision=head)
ops=[CommitOperationDelete(path_in_repo=n) for n in existing if n not in names and n!='.gitattributes']
ops += [CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(files)]
print('Uploading',len(files),'release files; parent',head,flush=True)
commit=api.create_commit(repo_id=repo,operations=ops,parent_commit=head,commit_message='Release Jet v6.1: full merged step-2000 continuation and expanded benchmark evidence',num_threads=2)
api.create_tag(repo,tag='v6.1.0',revision=commit.oid,tag_message='Jet full merged BF16 continuation, selected step 2000',exist_ok=True)
remote=list(api.list_repo_tree(repo,revision=commit.oid,recursive=True));assert {r.path for r in remote if hasattr(r,'size')}==names|{'.gitattributes'}
for r in remote:
 if getattr(r,'lfs',None):assert r.lfs.sha256==manifest['files'][r.path],r.path
receipt={'repo_id':repo,'revision':commit.oid,'tag':'v6.1.0','previous_revision':head,'previous_tag':'v6.0.0','format':'full merged BF16 model','remote_large_file_hashes_verified':True,'url':'https://huggingface.co/michaljach/jet'}
(HERE/'published.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
