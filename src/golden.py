"""Golden test vectors for ports of Jet to other runtimes (e.g. a browser build).

    jet-golden --base-model models/jet --out models/jet/golden.json

For each case it records everything a port has to reproduce: the rendered prompt text, its token ids,
the label strings and their token ids, the raw label logits, and the calibrated probabilities and typed
answer. Prompt text and token ids should match exactly; logits and probabilities only up to the port's
numeric precision (bf16 here), so compare those with a tolerance and check that the argmax agrees.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from evaluate import softmax
from format import SYSTEM_PROMPT, Question, label_token_ids, labels_for
from model import DEFAULT_BASE_MODEL, Jet, encode, prompt_text, summarize

# Edge cases the test set doesn't cover well: structured and non-ASCII states, labels past Z,
# the widest score scale, and a state long enough to be cut in the middle.
EDGE_CASES: list[tuple[str, Any, dict]] = [
    (
        "json_state",
        {"user": "Zoë", "plan": "pro", "tickets": [{"id": 17, "text": "Refund? Charged twice 💳"}], "active": True},
        {"type": "noul", "instructions": "Is the customer asking for money back?"},
    ),
    (
        "list_state",
        ["User: my build fails on ARM", "Agent: which compiler?", "User: clang 19, see log above"],
        {"type": "choice", "instructions": "What should the agent do next?",
         "criteria": {"ask_log": "ask for the full build log", "close": "close the ticket", "escalate": "hand to a human"}},
    ),
    (
        "non_ascii_state",
        "東京の配達が三日遅れています。至急対応してください！ Merci d'avance — très urgent.",
        {"type": "score", "instructions": "How urgent is this message?", "criteria": ["low", "medium", "high"]},
    ),
    (
        "two_letter_labels",
        "The invoice lists a 30-day payment term but the contract says 45 days.",
        {"type": "choice", "instructions": "Which category fits best?",
         "criteria": {f"cat_{i:02d}": f"category number {i}" for i in range(40)} | {"contract_mismatch": "terms differ between documents"}},
    ),
    (
        "ten_levels",
        "The patch fixes the crash but adds no tests and touches the payment path.",
        {"type": "score", "instructions": "How risky is merging this change?",
         "criteria": [f"risk level {i}" for i in range(10)]},
    ),
    (
        "truncated_state",
        "\n".join(f"log line {i}: request handled in {i % 97} ms, status {'500' if i == 2500 else '200'}" for i in range(5000)),
        {"type": "noul", "instructions": "Does the log contain a server error?"},
    ),
]

# Multi-question requests take the shared-state path (the state is encoded once).
DECIDE_CASES: list[tuple[Any, dict[str, dict]]] = [
    (
        "I was charged twice this month and nobody answers my emails.",
        {
            "topic": {"type": "choice", "instructions": "What is the primary issue?",
                      "criteria": {"billing": "billing or payment problem", "bug": "the product is broken"}},
            "severity": {"type": "score", "instructions": "How urgent is this?",
                         "criteria": ["routine", "handle today", "urgent", "critical, about to churn"]},
            "escalate": {"type": "noul", "instructions": "Escalate to a human immediately?"},
        },
    ),
    (
        {"tool": "shell", "command": "rm -rf ./build && make", "cwd": "/home/dev/project"},
        {
            "safe": {"type": "noul", "instructions": "Is this tool call safe to run without asking?"},
            "risk": {"type": "score", "instructions": "How destructive could this be?",
                     "criteria": ["harmless", "recoverable", "data loss", "system damage"]},
        },
    ),
]


def case(jet: Jet, name: str, state: Any, question: dict) -> dict:
    q = Question.from_dict(question)
    (logits,), _ = jet.raw_logits([(state, q)])
    probs = softmax(logits, jet.temperatures[q.type])
    return {
        "name": name,
        "state": state,
        "question": q.to_dict(),
        "prompt": prompt_text(jet.tokenizer, state, q, jet.max_state_tokens),
        "input_ids": encode(jet.tokenizer, state, q, jet.max_state_tokens),
        "labels": labels_for(jet.tokenizer, q),
        "label_token_ids": label_token_ids(jet.tokenizer, q),
        "logits": [round(float(z), 4) for z in logits],
        "temperature": jet.temperatures[q.type],
        "probabilities": [round(float(p), 6) for p in probs],
        "answer": summarize(q, probs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Write golden test vectors for ports of Jet.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", type=Path, default=Path("data/test.jsonl"))
    ap.add_argument("--per-type", type=int, default=10, help="test-set cases per question type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("golden.json"))
    args = ap.parse_args()

    jet = Jet(args.base_model, args.adapter)
    rows = [json.loads(line) for line in args.data.open()]
    rng = random.Random(args.seed)
    picked = []
    for qtype in ("choice", "score", "noul"):
        of_type = [r for r in rows if r["question"]["type"] == qtype]
        # Spread over sources, and make sure structured (dict) states are included.
        by_source: dict[str, list[dict]] = {}
        for r in of_type:
            by_source.setdefault(r["source"], []).append(r)
        pool = [rng.choice(rs) for rs in by_source.values()]
        dicts = [r for r in of_type if isinstance(r["state"], dict)]
        if dicts:
            pool.append(rng.choice(dicts))
        rest = [r for r in of_type if r not in pool]
        pool += rng.sample(rest, max(0, args.per_type - len(pool)))
        picked += [(f"{qtype}_{r['source']}_{i}", r["state"], r["question"]) for i, r in enumerate(pool[: args.per_type])]

    cases = [case(jet, name, state, q) for name, state, q in picked + EDGE_CASES]
    decide = []
    for state, questions in DECIDE_CASES:
        result = jet.decide(state, {name: Question.from_dict(q) for name, q in questions.items()})
        decide.append({"state": state, "questions": questions, "answers": result["answers"]})

    golden = {
        "model": args.base_model,
        "system_prompt": SYSTEM_PROMPT,
        "max_state_tokens": jet.max_state_tokens,
        "temperatures": jet.temperatures,
        "cases": cases,
        "decide": decide,
    }
    args.out.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases and {len(decide)} decide requests to {args.out}")


if __name__ == "__main__":
    main()
