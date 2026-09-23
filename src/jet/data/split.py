"""Merge example files and split into train / val / test.

Splits are grouped by question (source + instructions), so val/test questions
are ones the model never saw during training - that measures generalisation to
new decision tasks, which is the whole point of Jet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from jet.format import Question


def bucket(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge and split Jet example files.")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--holdout-source", nargs="*", default=[], help="sources that go entirely to test (zero-shot eval)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seen, splits = set(), {"train": [], "val": [], "test": []}
    for path in args.inputs:
        for line in path.open():
            ex = json.loads(line)
            Question.from_dict(ex["question"])
            dedupe = json.dumps([ex["state"], ex["question"]], sort_keys=True)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            # Public datasets reuse a handful of instructions, so group them by state instead: one
            # text asked several questions (e.g. stsb and stsb_scales) must not straddle splits.
            group = ex["question"]["instructions"] if ex["source"].startswith("distill:") else json.dumps(ex["state"], sort_keys=True)
            b = bucket(ex["source"] + "|" + group)
            if ex["source"] in args.holdout_source:
                split = "test"
            else:
                split = "test" if b < args.test else "val" if b < args.test + args.val else "train"
            splits[split].append(ex)

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        rng.shuffle(examples)
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name:>5}: {len(examples)}")


if __name__ == "__main__":
    main()


def extend_main() -> None:
    """Add new examples to an existing train split without touching val/test.

    Changing the public builders and re-running jet-split reshuffles every split, so the
    new model's test numbers would not be comparable with the old one's. This keeps the
    splits fixed and drops any new example whose state already appears in a held-out file.
    """
    ap = argparse.ArgumentParser(description="Append examples to a train split, excluding held-out states.")
    ap.add_argument("train", type=Path, help="existing train split")
    ap.add_argument("inputs", nargs="+", type=Path, help="new example files")
    ap.add_argument("--exclude", nargs="+", type=Path, required=True, help="held-out files whose states must stay unseen")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    held = {json.dumps(json.loads(line)["state"], sort_keys=True) for path in args.exclude for line in path.open()}
    rows = [json.loads(line) for line in args.train.open()]
    seen = {json.dumps([ex["state"], ex["question"]], sort_keys=True) for ex in rows}
    added = overlap = 0
    for path in args.inputs:
        for line in path.open():
            ex = json.loads(line)
            Question.from_dict(ex["question"])
            if json.dumps(ex["state"], sort_keys=True) in held:
                overlap += 1
                continue
            dedupe = json.dumps([ex["state"], ex["question"]], sort_keys=True)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            rows.append(ex)
            added += 1

    random.Random(args.seed).shuffle(rows)
    with args.out.open("w") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"added {added}, dropped {overlap} with held-out states; {args.out}: {len(rows)}")
