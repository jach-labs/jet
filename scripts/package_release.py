"""Package a checked local release with provenance and an explicit upload manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--check-log', type=Path, required=True)
    p.add_argument('--fusion-check', type=Path, required=True)
    args = p.parse_args()
    check = args.check_log.read_text()
    assert '36 golden cases:' in check and 'answers flipped 0' in check
    fusion = json.loads(args.fusion_check.read_text())
    assert fusion['cases'] == 36 and fusion['answer_flips'] == []
    meta = '---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-0.6B\nlibrary_name: mlx\nlanguage:\n- en\ntags:\n- decision-model\n- text-classification\n- mlx\n- qwen3\n- lora\n---\n\n'
    readme = Path('README.md').read_text()
    for prefix in ['deploy/', 'src/']:
        readme = readme.replace(']('+prefix, '](https://github.com/jach-labs/jet/blob/main/'+prefix)
    (args.model/'README.md').write_text(meta+readme)
    shutil.copy2('TRAINING_HISTORY.md', args.model/'TRAINING_HISTORY.md')
    shutil.copytree('docs/training', args.model/'docs/training', dirs_exist_ok=True)
    shutil.copy2(args.check_log, args.model/'export-validation.txt')
    provenance = {'adapter': args.adapter.name, 'adapter_sha256': digest(args.adapter/'adapters.safetensors'), 'fusion': 'Native MLX CUDA module.fuse(), 196 linear layers', 'fusion_validation': fusion, 'onnx_validation': check.strip()}
    (args.model/'export-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    files = {str(f.relative_to(args.model)): digest(f) for f in sorted(args.model.rglob('*')) if f.is_file() and f.name != 'release-manifest.json'}
    manifest = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'candidate':args.adapter.name, 'validation':{'onnx_golden_passed':True,'cases':36,'answer_flips':0,'reference_backend':'native MLX CUDA fused bf16'},'files':files}
    (args.model/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Packaged',len(files),'files')


if __name__ == '__main__':
    main()
