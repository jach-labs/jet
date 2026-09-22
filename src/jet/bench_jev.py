"""Benchmark the hosted Jev API on a Jet test set, for a head-to-head comparison.

    JEV_API_KEY=jv_live_... jet-bench-jev --data data/test.jsonl --limit 1500
    jet-bench-jev --score-only          # re-score saved responses without calling the API

Raw responses are appended to data/jev_responses.jsonl (keyed by row index) so a
re-run only calls the API for rows it doesn't have yet. Metrics use the same
functions as jet-eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jet.evaluate import metrics, print_table
from jet.format import Question

DEFAULT_URL = "https://jevtypesafeai.com/api/v1/decide"


def call(url: str, key: str, row: dict, retries: int = 4) -> dict:
    body = json.dumps({"model": "jev-latest", "state": row["state"], "questions": {"q": row["question"]}}).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return {"status": resp.status, "ms": (time.perf_counter() - started) * 1000, "body": json.loads(resp.read())}
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            return {"status": e.code, "ms": (time.perf_counter() - started) * 1000, "body": e.read().decode(errors="replace")}
    raise AssertionError("unreachable")


def distribution(q: Question, answer: dict) -> np.ndarray | None:
    """Pull a probability vector in Jet label order out of a Jev answer."""
    if q.type == "noul":
        p = answer.get("probability", answer.get("p_yes", answer.get("yes")))
        return None if p is None else np.array([1 - float(p), float(p)])
    probs = answer.get("probabilities", answer.get("distribution"))
    if isinstance(probs, dict):
        keys = q.keys if q.type == "choice" else list(probs)
        if q.type == "choice":
            return np.array([float(probs.get(k, 0.0)) for k in keys])
        return np.array([float(v) for v in probs.values()])
    if isinstance(probs, list):
        return np.array([float(p["probability"]) if isinstance(p, dict) else float(p) for p in probs])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the hosted Jev API on a Jet test set.")
    ap.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--url", default=os.environ.get("JEV_URL", DEFAULT_URL))
    ap.add_argument("--responses", type=Path, default=Path("data/jev_responses.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.data.open()][: args.limit]
    saved = {}
    if args.responses.exists():
        saved = {r["i"]: r for r in map(json.loads, args.responses.open())}

    todo = [i for i in range(len(rows)) if saved.get(i, {}).get("status") != 200]
    if todo and not args.score_only:
        key = os.environ.get("JEV_API_KEY") or sys.exit("set JEV_API_KEY")
        print(f"calling {args.url} for {len(todo)} rows")
        # Probe one row first so a bad key or unexpected schema fails fast.
        first = call(args.url, key, rows[todo[0]])
        print(json.dumps(first, indent=2)[:1500])
        if first["status"] != 200:
            sys.exit("probe request failed")
        with args.responses.open("a") as f:
            f.write(json.dumps({"i": todo[0], **first}) + "\n")
            with ThreadPoolExecutor(args.concurrency) as pool:
                for n, (i, res) in enumerate(zip(todo[1:], pool.map(lambda i: call(args.url, key, rows[i]), todo[1:])), 1):
                    f.write(json.dumps({"i": i, **res}) + "\n")
                    f.flush()
                    saved[i] = res
                    print(f"\r  {n + 1}/{len(todo)}", end="", flush=True)
        print()
        saved = {r["i"]: r for r in map(json.loads, args.responses.open())}

    probs, targets, sources, types, lat, unparsed = [], [], [], [], [], 0
    for i, row in enumerate(rows):
        res = saved.get(i)
        if not res or res["status"] != 200:
            continue
        q = Question.from_dict(row["question"])
        p = distribution(q, res["body"].get("answers", {}).get("q", {}))
        if p is None or len(p) != len(q.keys) or p.sum() <= 0:
            unparsed += 1
            continue
        probs.append(p / p.sum())
        targets.append(np.array(row["target"]))
        sources.append("distill (all)" if row["source"].startswith("distill:") else row["source"])
        types.append(row["question"]["type"])
        lat.append(res["ms"])
    if not probs:
        sys.exit(f"no parseable responses ({unparsed} unparsed); check the saved body format in {args.responses}")

    def group(keyfn):
        out = {}
        for k in sorted(set(keyfn)):
            ix = [j for j, v in enumerate(keyfn) if v == k]
            out[k] = metrics([probs[j] for j in ix], [targets[j] for j in ix])
        return out

    overall = metrics(probs, targets)
    print_table("Jev by source", group(sources))
    print_table("Jev by type", group(types))
    print_table("Jev overall", {"all": overall})
    lat_a = np.array(lat)
    print(f"\nlatency ms (network incl.): p50 {np.percentile(lat_a, 50):.0f}  p90 {np.percentile(lat_a, 90):.0f}; unparsed {unparsed}")
    if args.json:
        args.json.write_text(json.dumps({"overall": overall, "by_source": group(sources), "by_type": group(types),
                                         "latency_p50_ms": float(np.percentile(lat_a, 50))}, indent=2))


if __name__ == "__main__":
    main()
