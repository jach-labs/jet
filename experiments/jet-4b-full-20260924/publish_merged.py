"""Replace Jet main atomically with the verified full model; archive old release."""
import json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,ModelCard,CommitOperationAdd,CommitOperationDelete
P=Path('releases/jet-v6');api=HfApi();repo='michaljach/jet'
validation=json.loads((P/'merge-validation.json').read_text());assert validation['cases']==36 and validation['argmax_flips']==1 and validation['max_probability_difference']<.05
# The observed ordinal flip is explicitly disclosed in the release card.
ModelCard.load(P/'README.md').validate()
files=[p for p in sorted(P.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='release-manifest.json']
manifest={'version':'v6.0.0','name':'Jet','format':'complete merged BF16 Qwen3_5ForCausalLM','files':{str(p.relative_to(P)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files},'validation':{k:v for k,v in validation.items() if k not in ['records','wrapper_smoke']},'official_decision_index':None}
(P/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(P/'release-manifest.json')
head=api.model_info(repo).sha
if head!='25ccbd9e09c75643b3c2214e2b2522bec39171a7':raise RuntimeError('Remote changed; inspect before replacing')
api.create_tag(repo,tag='qwen3-0.6b-final',revision=head,tag_message='Final previous Qwen3-0.6B release',exist_ok=True)
existing=api.list_repo_files(repo,revision=head);new={str(p.relative_to(P)) for p in files}
ops=[CommitOperationDelete(path_in_repo=n) for n in existing if n not in new and n!='.gitattributes']
ops += [CommitOperationAdd(path_in_repo=str(p.relative_to(P)),path_or_fileobj=str(p)) for p in files]
result=api.create_commit(repo_id=repo,operations=ops,parent_commit=head,commit_message='Release Jet v6: full merged Qwen3.5-4B BF16 weights and native typed inference',num_threads=2)
api.create_tag(repo,tag='v6.0.0',revision=result.oid,tag_message='Jet full merged model; step 3750')
# Verify all remote large-file hashes and the complete runtime file set.
remote=list(api.list_repo_tree(repo,revision=result.oid,recursive=True));assert {r.path for r in remote if hasattr(r,'size')}==new|{'.gitattributes'}
for r in remote:
 if getattr(r,'lfs',None):assert r.lfs.sha256==manifest['files'][r.path],r.path
record={'repo_id':repo,'revision':result.oid,'tag':'v6.0.0','url':'https://huggingface.co/michaljach/jet','previous_release_tag':'qwen3-0.6b-final','remote_large_file_hashes_verified':True}
Path(__file__).with_name('published-merged-release.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
