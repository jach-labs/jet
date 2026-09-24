"""Upload a validated local Jet release to its existing Hugging Face model repo.

Uploads model files only; does not create a Space or select hosting hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--repo', default='michaljach/jet')
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.model / 'release-manifest.json').read_text())
    assert manifest['validation']['onnx_golden_passed'] is True
    for name, digest in manifest['files'].items():
        with (args.model / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    api = HfApi()
    parent = api.model_info(args.repo).sha
    # Explicit allowlist; atomic commit also replaces any old sharding index.
    names = list(manifest['files']) + ['release-manifest.json']
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(args.model / name)) for name in names]
    result = api.create_commit(repo_id=args.repo, repo_type='model', operations=operations, parent_commit=parent,
                               commit_message='Release Jet V4 transfer checkpoint with calibrated bf16 and q8 ONNX weights')
    receipt = {'repo': args.repo, 'previous_revision': parent, 'revision': result.oid, 'url': result.commit_url}
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(result.commit_url)


if __name__ == '__main__':
    main()
