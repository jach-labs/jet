"""Prepare, audit, and report Decision Index diagnostics without training on them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from decision_index.engines import Unsupported
from decision_index.scoring.index import score_panel
from decision_index.scoring.report import benchmark_summary, load_results
from decision_index.suite.io import Suite, atomic_json, read_jsonl, sha256_file
from decision_index.suite.sample import stratified_sample

from decision_index_engine import load_assets, prepare_request


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(suite, out, n, allow_partial=False):
    verification = suite.verify(strict=not allow_partial)
    # Even partial rebuilds must use the published exclusions.
    if not verification.get("exclusions_match"):
        raise ValueError("Missing or mismatched official exclusions file")
    out.mkdir(parents=True, exist_ok=True)
    tracks = defaultdict(list)
    for row in suite.rows(apply_exclusions=True):
        key = row["_evaluation"]["track"]
        # Deterministic, label-independent selection, independent of file order.
        rank = hashlib.sha256(row["_evaluation"]["run_id"].encode()).hexdigest()
        tracks[key].append((rank, row))
        tracks[key] = sorted(tracks[key], key=lambda x: x[0])[:2]
    smoke = [row for key in sorted(tracks) for _, row in tracks[key]]
    write_rows(out / "compatibility.jsonl.gz", smoke)
    sample = stratified_sample(suite, n, out / "diagnostic.jsonl.gz")
    report = {"verification": verification, "full_frozen_suite": verification["match"],
              "compatibility_requests": len(smoke), "tracks": len(tracks),
              "diagnostic": sample, "note": "Compatibility is two requests per available track; diagnostic sampling keeps complete source groups. Neither is a leaderboard score."}
    atomic_json(out / "preparation.json", report)
    return report


def audit(rows_path, out):
    path, tokenizer, config = load_assets()
    limits = [4096, 6144, 8192, 16384, 32768, config["max_position_embeddings"]]
    stats = defaultdict(lambda: {"requests": 0, "fields": 0, "lengths": [],
                                 "options": Counter(), "questions_per_request": Counter(),
                                 "supported_requests": Counter(), "capacity_failures": Counter()})
    groups = defaultdict(lambda: {limit: True for limit in limits})
    for i, row in enumerate(read_jsonl(rows_path)):
        e = row["_evaluation"]
        s = stats[e["dataset"]]
        s["requests"] += 1
        s["fields"] += len(row["questions"])
        s["questions_per_request"][len(row["questions"])] += 1
        for q in row["questions"].values():
            s["options"][len(q["criteria"])] += 1
        try:
            prepared = prepare_request(tokenizer, row["state"], row["questions"], 10**12)
            lengths = [len(p[2]) for p in prepared]
            s["lengths"].extend(lengths)
            maximum = max(lengths)
        except Unsupported as exc:
            s["capacity_failures"][str(exc)] += 1
            maximum = float("inf")
        group = groups[(e["dataset"], e["group_id"])]
        for limit in limits:
            supported = maximum <= limit
            s["supported_requests"][limit] += int(supported)
            group[limit] &= supported
        if (i + 1) % 500 == 0:
            print(f"Audited {i + 1} requests", flush=True)
    for name, s in stats.items():
        lengths = s.pop("lengths")
        s["prompt_tokens"] = dict(zip(["p50", "p95", "max"], map(float, np.percentile(lengths, [50, 95, 100])))) if lengths else None
        selected = [g for (dataset, _), g in groups.items() if dataset == name]
        s["source_groups"] = len(selected)
        s["supported_complete_groups"] = {limit: sum(g[limit] for g in selected) for limit in limits}
    report = {"rows_sha256": sha256_file(rows_path),
              "rows_uncompressed_sha256": sha256_file(rows_path, gunzip=Path(rows_path).suffix == ".gz"),
              "model_revision": path.name,
              "scope": "Only the supplied rows; token capacity, not measured accuracy or validated long-context quality.",
              "limits": limits, "benchmarks": dict(stats)}
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, report)
    return report


class SampleSuite:
    """Read exactly the diagnostic rows, with no invented full-suite denominator."""
    def __init__(self, rows_path):
        self.rows_path = rows_path

    def rows(self, apply_exclusions=False):
        yield from read_jsonl(self.rows_path)


def report(rows_path, results_path, out):
    suite = SampleSuite(rows_path)
    results = load_results(results_path)
    summary = benchmark_summary(suite, results, "jet")
    panel = score_panel(suite, results)
    planned = {r["_evaluation"]["run_id"] for r in suite.rows()}
    present = planned & results.keys()
    environment_path = Path(results_path).with_name("environment.json")
    data = {"scope": "Diagnostic subset only. Not an official Decision Index or full-suite estimate.",
            "rows_sha256": sha256_file(rows_path), "results_sha256": sha256_file(results_path),
            "planned_requests": len(planned), "processed_requests": len(present),
            "pending_requests": len(planned - present),
            "complete_for_supplied_rows": planned == present,
            "environment": json.loads(environment_path.read_text()) if environment_path.exists() else None,
            "summary": summary, "panel_metrics_on_sample": panel}
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, data)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--allow-partial", action="store_true", help="Explicitly permit a nonmatching partial rebuilt corpus")
    p = sub.add_parser("audit")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "prepare":
        if args.n < 1:
            ap.error("--n must be positive")
        data = prepare(Suite(args.suite_dir), args.out, args.n, args.allow_partial)
        print(json.dumps(data, indent=2))
    elif args.command == "audit":
        audit(args.rows, args.out)
        print(f"Wrote {args.out}")
    else:
        report(args.rows, args.results, args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
