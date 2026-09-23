"""Upload explicitly selected deployment files; never choose paid hardware.

Run using a Python environment with huggingface_hub installed:
  python deploy/huggingface/deploy.py static --repo michaljach/jet
  python deploy/huggingface/deploy.py api --repo michaljach/jet-api
  python deploy/huggingface/deploy.py zerogpu --repo michaljach/jet
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "src"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["static", "api", "zerogpu"])
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    api = HfApi()
    sdk = {"static": "static", "api": "docker", "zerogpu": "gradio"}[args.kind]
    hardware = {"api": "cpu-basic", "zerogpu": "zero-a10g"}.get(args.kind)
    api.create_repo(args.repo, repo_type="space", space_sdk=sdk,
                    **({"space_hardware": hardware} if hardware else {}),
                    private=False, exist_ok=True)
    if api.space_info(args.repo).sdk != sdk:
        raise RuntimeError("Existing Space SDK differs; refusing to overwrite it")
    operations = []
    if args.kind == "static":
        existing = set(api.list_repo_files(args.repo, repo_type="space"))
        for name in ("jet-client.js", "jet-core.js", "jet-worker.js"):
            if name in existing:
                operations.append(CommitOperationDelete(path_in_repo=name))
    def add(file, target):
        operations.append(CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(file)))
    for file in sorted((ROOT / args.kind).iterdir()):
        if file.is_file(): add(file, file.name)
    for file in sorted((ROOT / "api").iterdir()):
        if file.is_file() and args.kind == "static": add(file, "api/" + file.name)
    backend = "torch_model.py" if args.kind == "zerogpu" else "onnx_model.py"
    for name in ["format.py", "inference.py", backend]:
        add(SOURCE / name, ("api/" if args.kind == "static" else "") + name)
    if args.kind == "static":
        add(ROOT / "README.md", "DEPLOYMENT.md")
        for file in sorted((ROOT / "zerogpu").iterdir()):
            if file.is_file(): add(file, "zerogpu/" + file.name)
        for name in ["format.py", "inference.py", "torch_model.py"]:
            add(SOURCE / name, "zerogpu/" + name)
    print(api.create_commit(repo_id=args.repo, repo_type="space", operations=operations,
                            commit_message=f"Deploy Jet {args.kind} with pinned model"))
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
