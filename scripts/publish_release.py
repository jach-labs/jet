"""Publish a validated release folder to the Hugging Face model repository's main branch.

    uv run python scripts/publish_release.py --version v6.0.1 --dry-run
    uv run python scripts/publish_release.py --version v6.0.1 --receipt releases/jet-v6/publish-receipt.json

Copies src/format.py and src/inference.py into the folder, writes release-manifest.json
with per-file SHA256s, then replaces main in one commit: every file in the folder is
uploaded and remote files not in it are deleted. Creates no tags or branches. After the
commit, remote file names and large-file hashes are checked against the manifest.
Generalized from experiments/jet-4b-full-20260924/publish_merged.py (the v6 upload).
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, ModelCard

from release_files import DEFAULT_RELEASE, sync_code

KEEP_REMOTE = {'.gitattributes'}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    ap.add_argument('--repo', default='michaljach/jet')
    ap.add_argument('--version', required=True)
    ap.add_argument('--message', default=None, help='commit message')
    ap.add_argument('--dry-run', action='store_true', help='print the commit without writing anything remote')
    ap.add_argument('--receipt', type=Path)
    args = ap.parse_args()

    P = args.release
    validation = json.loads((P / 'merge-validation.json').read_text())
    ModelCard.load(P / 'README.md').validate()
    shards = set(json.loads((P / 'model.safetensors.index.json').read_text())['weight_map'].values())
    missing = sorted(n for n in shards | {'tokenizer.json'} if not (P / n).is_file())
    if missing and not args.dry_run:
        raise SystemExit(f'{P} is missing {missing}; run scripts/merge_release.py first')
    sync_code(P)

    files = [p for p in sorted(P.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name not in ('release-manifest.json', 'publish-receipt.json')]
    architecture = json.loads((P / 'config.json').read_text())['architectures'][0]
    manifest = {
        'version': args.version, 'name': 'Jet', 'format': f'complete merged BF16 {architecture}',
        'files': {str(p.relative_to(P)): digest(p) for p in files},
        'validation': {k: v for k, v in validation.items() if k not in ('records', 'wrapper_smoke')},
        'official_decision_index': None,
    }
    files.append(P / 'release-manifest.json')
    new = {str(p.relative_to(P)) for p in files}

    api = HfApi()
    head = api.model_info(args.repo).sha
    existing = api.list_repo_files(args.repo, revision=head)
    deletes = sorted(n for n in existing if n not in new and n not in KEEP_REMOTE)
    print(f'{args.repo}@{head[:10]}: upload {len(files)} files, delete {len(deletes)}: {deletes}')
    if args.dry_run:
        return
    (P / 'release-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    ops = [CommitOperationDelete(path_in_repo=n) for n in deletes]
    ops += [CommitOperationAdd(path_in_repo=str(p.relative_to(P)), path_or_fileobj=str(p)) for p in files]

    result = api.create_commit(repo_id=args.repo, operations=ops, parent_commit=head, num_threads=2,
                               commit_message=args.message or f'Release Jet {args.version}')
    remote = list(api.list_repo_tree(args.repo, revision=result.oid, recursive=True))
    if {r.path for r in remote if hasattr(r, 'size')} != new | KEEP_REMOTE:
        raise RuntimeError('remote file set differs from the release folder')
    for r in remote:
        if getattr(r, 'lfs', None) and r.lfs.sha256 != manifest['files'][r.path]:
            raise RuntimeError(f'remote hash mismatch: {r.path}')
    receipt = {'repo': args.repo, 'version': args.version, 'previous_revision': head,
               'revision': result.oid, 'url': result.commit_url, 'remote_large_file_hashes_verified': True}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
