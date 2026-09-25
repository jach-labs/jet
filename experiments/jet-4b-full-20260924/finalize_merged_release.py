"""Verify published Jet and remove the accidentally separate release repository."""
import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
HERE=Path(__file__).resolve().parent
record=json.loads((HERE/'published-merged-release.json').read_text())
api=HfApi();repo=record['repo_id'];revision=record['revision']
assert repo=='michaljach/jet' and api.model_info(repo).sha==revision
assert api.model_info(repo,revision='v6.0.0').sha==revision
for name in ['config.json','calibration.json','README.md','jet.py','runtime.py','model.safetensors.index.json']:
 remote=Path(hf_hub_download(repo,name,revision=revision)).read_bytes()
 assert remote==(Path('releases/jet-v6')/name).read_bytes(),name
assert api.model_info(repo,revision='qwen3-0.6b-final').sha=='25ccbd9e09c75643b3c2214e2b2522bec39171a7'
files=api.list_repo_files(repo,revision=revision)
assert 'adapter_config.json' not in files and 'adapter_model.safetensors' not in files
assert not any(f.startswith('onnx/') for f in files)
# This is exactly the separate repository created by this task, not another model.
separate='michaljach/jet-4b'
assert api.model_info(separate).sha=='2af8c6e3acc7da703df79f042a72ff173792b3a4'
api.delete_repo(separate,repo_type='model')
record.update(separate_repository_deleted=separate,remote_runtime_files_verified=True,files=files)
(HERE/'published-merged-release.json').write_text(json.dumps(record,indent=2)+'\n')
old=HERE/'published-release.json';previous=json.loads(old.read_text());previous['superseded_by']=record['url'];previous['repository_deleted']=True;old.write_text(json.dumps(previous,indent=2)+'\n')
print(json.dumps({'url':record['url'],'revision':revision,'tag':record['tag'],'separate_repo_deleted':True,'files':len(files)}))
