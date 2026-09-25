"""Archive legacy experiment files and keep the model's main branch focused."""
import hashlib,json,re
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd,CommitOperationDelete
api=HfApi();repo='michaljach/jet';head=api.model_info(repo).sha
expected='ee7d33ecae2b8bee30891657dd57c92d34e0ce53'
if head!=expected:raise RuntimeError('Model repository changed; review before cleaning')
files=api.list_repo_files(repo,revision=head)
remove=[f for f in files if f.startswith('docs/training/')]
assert remove and all(not f.endswith(('.safetensors','.onnx')) for f in remove)
archive=f'https://huggingface.co/{repo}/blob/{head}/'
updates={}
for f in ['README.md','TRAINING_HISTORY.md']:
 text=Path(hf_hub_download(repo,f,revision=head)).read_text()
 text=re.sub(r'\]\((docs/training/[^)]+)\)',lambda m:']('+archive+m.group(1)+')',text)
 if f=='README.md':
  text=text.replace('# Jet\n','# Jet\n\nThe new **[Jet 4B release](https://huggingface.co/michaljach/jet-4b)** is available separately. This repository contains the Qwen3-0.6B model.\n',1)
 text+='\nDetailed experiment files are preserved in the [training archive](https://huggingface.co/michaljach/jet/tree/training-archive-20260924/docs/training).\n'
 updates[f]=text.encode()
manifest=json.loads(Path(hf_hub_download(repo,'release-manifest.json',revision=head)).read_text())
manifest['files']={k:v for k,v in manifest['files'].items() if k not in remove}
for f,b in updates.items():manifest['files'][f]=hashlib.sha256(b).hexdigest()
manifest['training_archive']={'revision':head,'tag':'training-archive-20260924','removed_from_main':len(remove)}
updates['release-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
# Tag the exact old revision before touching the default branch.
api.create_tag(repo,tag='training-archive-20260924',revision=head,tag_message='Complete experiment archive before Files cleanup',exist_ok=True)
ops=[CommitOperationDelete(path_in_repo=f) for f in remove]
ops.extend(CommitOperationAdd(path_in_repo=f,path_or_fileobj=b) for f,b in updates.items())
r=api.create_commit(repo_id=repo,operations=ops,parent_commit=head,commit_message='Archive detailed training artifacts and simplify model Files view')
remaining=api.list_repo_files(repo,revision=r.oid)
assert set(remaining)==set(files)-set(remove)
for f in ['model.safetensors','onnx/model_q8.onnx','config.json','calibration.json','tokenizer.json']:
 assert f in remaining
Path(__file__).with_name('hub-cleanup.json').write_text(json.dumps({'repo':repo,'before':head,'after':r.oid,'removed':remove,'remaining':remaining,'archive_tag':'training-archive-20260924'},indent=2)+'\n')
print(json.dumps({'revision':r.oid,'archived_files':len(remove),'remaining_files':len(remaining)}))
