"""Rebuild the eight public diagnostic benchmarks using pinned upstream code.

Run: uv run --extra benchmark python scripts/rebuild_index_diagnostic.py
This is NOT the full frozen suite. Never submit it as a leaderboard run.
"""
import argparse
import json
from pathlib import Path

from decision_index.suite.build.acquire import acquire, http_fetch
from decision_index.suite.build.layout import Layout
from decision_index.suite.build.rebuild import main as rebuild

KIT_REVISION = "52a698928a9ae5bdf16b75687c903871db29c6e5"
BENCHMARKS = [11, 30, 40, 41, 42, 43, 44, 50]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("artifacts/decision-index/rebuild"))
    args = ap.parse_args()
    layout = Layout(args.work)
    # The upstream builders group ContractNLI/NLI4CT, GSM8K/CRUXEval,
    # and VAST/CLadder. Acquire both members even for a partial rebuild.
    acquire(layout, BENCHMARKS)
    http_fetch(
        layout.repos / "habermas_machine/hm_all_candidate_comparisons.parquet",
        "https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet",
        "7cf8d5ce3fce8853b36f0ffe1158424f7813867e422e313db4df0a5f9e03e4a4",
    )
    result = rebuild(args.work, only=BENCHMARKS, skip_download=True)
    out = Path(result["out"])
    http_fetch(out / "excluded-questions.json",
               f"https://raw.githubusercontent.com/apolinario/decision-index/{KIT_REVISION}/hub/excluded-questions.json",
               "331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
