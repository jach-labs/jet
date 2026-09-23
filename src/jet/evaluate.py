"""Evaluate and calibrate a Jet model.

    jet-eval --adapter adapters/jet --data data/test.jsonl
    jet-eval --data data/test.jsonl                # untrained base model, as a baseline
    jet-calibrate --adapter adapters/jet --data data/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from jet.format import Question
from jet.model import DEFAULT_BASE_MODEL, Jet

TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))


def load_data(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    return rows[:limit] if limit else rows


def adapter_base(adapter: str | None, fallback: str) -> str:
    if adapter and (cfg := Path(adapter) / "adapter_config.json").exists():
        return json.loads(cfg.read_text()).get("base_model", fallback)
    return fallback


def collect_logits(jet: Jet, rows: list[dict], batch_size: int) -> tuple[list[np.ndarray], float]:
    """Raw label logits for every row, batched by prompt length. Returns (logits, ms per question)."""
    items = [(r["state"], Question.from_dict(r["question"])) for r in rows]
    order = sorted(range(len(items)), key=lambda i: len(str(items[i][0])))
    logits: list[np.ndarray | None] = [None] * len(items)
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        out, _ = jet.raw_logits([items[i] for i in idx])
        for i, z in zip(idx, out):
            logits[i] = z
        print(f"\r  {min(start + batch_size, len(order))}/{len(order)}", end="", flush=True)
    print()
    return logits, (time.perf_counter() - started) * 1000 / max(len(items), 1)


def softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max())
    return e / e.sum()


def nll(logits: list[np.ndarray], targets: list[np.ndarray], t: float) -> float:
    return float(np.mean([-(tg * np.log(np.clip(softmax(z, t), 1e-12, 1))).sum() for z, tg in zip(logits, targets)]))


def metrics(probs: list[np.ndarray], targets: list[np.ndarray], bins: int = 10) -> dict[str, float]:
    top = np.array([p.max() for p in probs])
    hit = np.array([p.argmax() == t.argmax() for p, t in zip(probs, targets)], dtype=float)
    # ECE over the model's top prediction, scored against the (possibly soft) target's argmax.
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        sel = (top > lo) & (top <= lo + 1 / bins)
        if sel.any():
            ece += sel.mean() * abs(top[sel].mean() - hit[sel].mean())
    return {
        "n": len(probs),
        "acc": float(hit.mean()),
        "nll": float(np.mean([-(t * np.log(np.clip(p, 1e-12, 1))).sum() for p, t in zip(probs, targets)])),
        "brier": float(np.mean([((p - t) ** 2).sum() for p, t in zip(probs, targets)])),
        "ece": float(ece),
    }


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), for Spearman correlation."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):
        tie = x == v
        ranks[tie] = ranks[tie].mean()
    return ranks


def ordinal_metrics(probs: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    """Score questions: compare expected levels, normalized to 0-1 so scales of any size mix.

    mae: mean |E[pred] - E[target]| on the 0-1 scale; within1: argmax level within one of the
    target's expected level; spearman: rank correlation of expected scores.
    """
    pred = np.array([np.dot(np.arange(len(p)), p) / (len(p) - 1) for p in probs])
    true = np.array([np.dot(np.arange(len(t)), t) / (len(t) - 1) for t in targets])
    within = [abs(int(p.argmax()) - np.dot(np.arange(len(t)), t)) <= 1 for p, t in zip(probs, targets)]
    spearman = float(np.corrcoef(rank(pred), rank(true))[0, 1]) if len(pred) > 2 and true.std() > 0 else float("nan")
    return {"mae": float(np.abs(pred - true).mean()), "within1": float(np.mean(within)), "spearman": spearman}


def print_table(title: str, groups: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'':<42}{'n':>6}{'acc':>8}{'nll':>8}{'brier':>8}{'ece':>8}{'mae':>8}{'±1':>8}{'rho':>8}")
    for name, m in sorted(groups.items()):
        ordinal = "".join(f"{m[k]:>8.3f}" for k in ("mae", "within1", "spearman") if k in m)
        print(f"{name[:41]:<42}{m['n']:>6}{m['acc']:>8.3f}{m['nll']:>8.3f}{m['brier']:>8.3f}{m['ece']:>8.3f}{ordinal}")


def main_eval() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Jet model.")
    ap.add_argument("--adapter", default=None, help="adapter dir; omit to evaluate the base model")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", type=Path, default=None, help="also write metrics as JSON")
    args = ap.parse_args()

    jet = Jet(adapter_base(args.adapter, args.base_model), args.adapter)
    rows = load_data(args.data, args.limit)
    logits, ms = collect_logits(jet, rows, args.batch_size)
    probs = [softmax(z, jet.temperatures[r["question"]["type"]]) for z, r in zip(logits, rows)]
    targets = [np.array(r["target"]) for r in rows]

    by_source, by_type = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        source = "distill (all)" if r["source"].startswith("distill:") else r["source"]
        by_source[source].append(i)
        by_type[r["question"]["type"]].append(i)

    def group(ix: list[int]) -> dict[str, float]:
        m = metrics([probs[i] for i in ix], [targets[i] for i in ix])
        scored = [i for i in ix if rows[i]["question"]["type"] == "score"]
        if len(scored) == len(ix):
            m |= ordinal_metrics([probs[i] for i in scored], [targets[i] for i in scored])
        return m

    overall = metrics(probs, targets)
    print_table("by source", {k: group(v) for k, v in by_source.items()})
    print_table("by type", {k: group(v) for k, v in by_type.items()})
    print_table("overall", {"all": overall})
    print(f"\n{ms:.1f} ms/question (batched, batch size {args.batch_size}); temperatures {jet.temperatures}")
    if args.json:
        args.json.write_text(json.dumps({
            "overall": overall,
            "by_source": {k: group(v) for k, v in by_source.items()},
            "by_type": {k: group(v) for k, v in by_type.items()},
            "ms_per_question": ms,
        }, indent=2))


def main_calibrate() -> None:
    ap = argparse.ArgumentParser(description="Fit per-type softmax temperatures on held-out data.")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data", type=Path, default=Path("data/val.jsonl"))
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    jet = Jet(adapter_base(args.adapter, args.base_model), args.adapter)
    rows = load_data(args.data, args.limit)
    logits, _ = collect_logits(jet, rows, args.batch_size)

    temperatures = {}
    for qtype in ("choice", "score", "noul"):
        ix = [i for i, r in enumerate(rows) if r["question"]["type"] == qtype]
        if not ix:
            temperatures[qtype] = 1.0
            continue
        zs, ts = [logits[i] for i in ix], [np.array(rows[i]["target"]) for i in ix]
        scores = [nll(zs, ts, t) for t in TEMPERATURE_GRID]
        best = float(TEMPERATURE_GRID[int(np.argmin(scores))])
        temperatures[qtype] = round(best, 4)
        print(f"{qtype:>6}: n={len(ix)} T={best:.3f} nll {nll(zs, ts, 1.0):.4f} -> {min(scores):.4f}")
    (Path(args.adapter) / "calibration.json").write_text(json.dumps(temperatures, indent=2))
    print(f"wrote {Path(args.adapter) / 'calibration.json'}")
