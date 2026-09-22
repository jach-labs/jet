"""Reframe public datasets into Jet examples (choice / score / noul).

Each example is {"state", "question", "target", "source"} where target is a
probability distribution over the question's labels. Augmentations (option
shuffling, option subsets, paraphrased instructions, multiclass -> yes/no)
force the model to read the criteria instead of memorising dataset labels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset

Example = dict[str, Any]


def rows(name: str, config: str | None, split: str, n: int, seed: int) -> list[dict]:
    ds = load_dataset(name, config, split=split).shuffle(seed=seed)
    return list(ds.select(range(min(n, len(ds)))))


def choice_example(
    rng: random.Random,
    state: Any,
    instructions: list[str],
    options: dict[str, str],
    gold: str,
    source: str,
    max_options: int | None = None,
) -> Example:
    keys = list(options)
    if max_options and len(keys) > max_options:
        others = [k for k in keys if k != gold]
        keys = [gold] + rng.sample(others, rng.randint(1, max_options - 1))
    rng.shuffle(keys)
    return {
        "state": state,
        "question": {"type": "choice", "instructions": rng.choice(instructions), "criteria": {k: options[k] for k in keys}},
        "target": [1.0 if k == gold else 0.0 for k in keys],
        "source": source,
    }


def noul_example(state: Any, instructions: str, p_yes: float, source: str) -> Example:
    return {
        "state": state,
        "question": {"type": "noul", "instructions": instructions},
        "target": [1.0 - p_yes, p_yes],
        "source": source,
    }


def score_example(state: Any, instructions: str, levels: list[str], value: float, source: str) -> Example:
    """value is a (possibly fractional) level index; split its mass between neighbours."""
    lo = int(value)
    frac = value - lo
    target = [0.0] * len(levels)
    target[lo] = 1.0 - frac
    if frac:
        target[lo + 1] = frac
    return {
        "state": state,
        "question": {"type": "score", "instructions": instructions, "criteria": levels},
        "target": target,
        "source": source,
    }


def multiclass_to_noul(rng: random.Random, state: Any, template: str, options: dict[str, str], gold: str, source: str) -> Example:
    """Balanced yes/no: half the time ask about the true class, half about a wrong one."""
    key = gold if rng.random() < 0.5 else rng.choice([k for k in options if k != gold])
    return noul_example(state, template.format(options[key]), 1.0 if key == gold else 0.0, source)


def pretty(label: str) -> str:
    return label.replace("_", " ")


# ---------------------------------------------------------------------------
# Builders. Each takes (rng, n) and yields examples.
# ---------------------------------------------------------------------------

TOPIC_Q = ["What is this text mainly about?", "Which category best fits this text?", "Classify the topic of this text."]
INTENT_Q = ["What does the user want?", "Which intent best matches this message?", "Route this request to the right intent."]


def boolq(rng: random.Random, n: int) -> Iterator[Example]:
    for r in rows("google/boolq", None, "train", n, rng.randint(0, 10**6)):
        yield noul_example(r["passage"], r["question"].capitalize() + "?", float(r["answer"]), "boolq")


def mnli(rng: random.Random, n: int) -> Iterator[Example]:
    options = {
        "entailment": "the hypothesis is definitely true given the premise",
        "neutral": "the hypothesis might be true or false; the premise does not settle it",
        "contradiction": "the hypothesis is definitely false given the premise",
    }
    names = list(options)
    for r in rows("nyu-mll/glue", "mnli", "train", n, rng.randint(0, 10**6)):
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        gold = names[r["label"]]
        if rng.random() < 0.6:
            yield choice_example(rng, state, ["How does the hypothesis relate to the premise?"], options, gold, "mnli")
        else:
            yield noul_example(state, "Does the premise imply the hypothesis?", float(gold == "entailment"), "mnli")


def stsb(rng: random.Random, n: int) -> Iterator[Example]:
    levels = [
        "completely unrelated",
        "different topics but share a few details",
        "not equivalent but on the same topic",
        "roughly equivalent, some important details differ",
        "mostly equivalent, only minor details differ",
        "completely equivalent in meaning",
    ]
    for r in rows("nyu-mll/glue", "stsb", "train", n, rng.randint(0, 10**6)):
        state = {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}
        yield score_example(state, "How similar in meaning are the two sentences?", levels, float(r["label"]), "stsb")


def emotion(rng: random.Random, n: int) -> Iterator[Example]:
    options = {
        "sadness": "the writer feels sad, down or hurt",
        "joy": "the writer feels happy, content or pleased",
        "love": "the writer feels love, affection or tenderness",
        "anger": "the writer feels angry, irritated or resentful",
        "fear": "the writer feels afraid, anxious or nervous",
        "surprise": "the writer feels surprised or amazed",
    }
    names = list(options)
    for r in rows("dair-ai/emotion", "split", "train", n, rng.randint(0, 10**6)):
        gold = names[r["label"]]
        if rng.random() < 0.7:
            yield choice_example(rng, r["text"], ["Which emotion does the writer express?", "What is the dominant emotion?"], options, gold, "emotion")
        else:
            yield multiclass_to_noul(rng, r["text"], "Does the writer express this: {}?", options, gold, "emotion")


def ag_news(rng: random.Random, n: int) -> Iterator[Example]:
    options = {
        "world": "world news, politics, international affairs",
        "sports": "sports, athletes, games and competitions",
        "business": "business, markets, companies and the economy",
        "scitech": "science and technology",
    }
    names = list(options)
    for r in rows("fancyzhx/ag_news", None, "train", n, rng.randint(0, 10**6)):
        gold = names[r["label"]]
        if rng.random() < 0.7:
            yield choice_example(rng, r["text"], TOPIC_Q, options, gold, "ag_news")
        else:
            yield multiclass_to_noul(rng, r["text"], "Is this article about {}?", options, gold, "ag_news")


def dbpedia(rng: random.Random, n: int) -> Iterator[Example]:
    names = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building",
             "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
    descs = ["a company or business", "a school, college or university", "an artist or musician", "an athlete",
             "a politician or office holder", "a vehicle, ship, aircraft or other means of transportation", "a building or structure",
             "a natural place such as a mountain, river or lake", "a village or small settlement", "an animal", "a plant",
             "a music album", "a film", "a book or other written work"]
    options = dict(zip(names, descs))
    for r in rows("fancyzhx/dbpedia_14", None, "train", n, rng.randint(0, 10**6)):
        state = f"{r['title']}\n{r['content'].strip()}"
        yield choice_example(rng, state, ["What kind of entity is described?", "Which category does this entity belong to?"], options, names[r["label"]], "dbpedia", max_options=rng.choice([6, 10, 14]))


def banking77(rng: random.Random, n: int) -> Iterator[Example]:
    ds = load_dataset("mteb/banking77", split="train")
    options = {name: pretty(name) for name in sorted(set(ds["label_text"]))}
    for r in ds.shuffle(seed=rng.randint(0, 10**6)).select(range(min(n, len(ds)))):
        if rng.random() < 0.8:
            yield choice_example(rng, r["text"], INTENT_Q, options, r["label_text"], "banking77", max_options=rng.choice([8, 20, 40, 77]))
        else:
            yield multiclass_to_noul(rng, r["text"], "Is the customer asking about: {}?", options, r["label_text"], "banking77")


def massive_intent(rng: random.Random, n: int) -> Iterator[Example]:
    ds = load_dataset("SetFit/amazon_massive_intent_en-US", split="train")
    options = {name: pretty(name) for name in sorted(set(ds["label_text"]))}
    for r in ds.shuffle(seed=rng.randint(0, 10**6)).select(range(min(n, len(ds)))):
        yield choice_example(rng, r["text"], INTENT_Q, options, r["label_text"], "massive", max_options=rng.choice([5, 15, 30, 60]))


def yelp(rng: random.Random, n: int) -> Iterator[Example]:
    levels = ["very negative", "negative", "mixed or neutral", "positive", "very positive"]
    for r in rows("Yelp/yelp_review_full", None, "train", n, rng.randint(0, 10**6)):
        if rng.random() < 0.7:
            yield score_example(r["text"], "How positive is this review?", levels, float(r["label"]), "yelp")
        elif r["label"] != 2:
            positive = r["label"] > 2
            if rng.random() < 0.5:
                yield noul_example(r["text"], "Is the customer satisfied?", float(positive), "yelp")
            else:
                yield noul_example(r["text"], "Is this a complaint?", float(not positive), "yelp")


def civil_comments(rng: random.Random, n: int) -> Iterator[Example]:
    # toxicity = fraction of raters who marked the comment toxic -> a natural soft label.
    # Oversample toxic comments; they are ~8% of the data.
    ds = load_dataset("google/civil_comments", split="train").shuffle(seed=rng.randint(0, 10**6)).select(range(n * 8))
    toxic = [r for r in ds if r["toxicity"] >= 0.3][: n // 2]
    clean = [r for r in ds if r["toxicity"] < 0.3][: n - len(toxic)]
    for r in toxic + clean:
        field, question = rng.choice([
            ("toxicity", "Is this comment toxic, rude or disrespectful?"),
            ("insult", "Does this comment insult someone?"),
            ("threat", "Does this comment contain a threat?"),
            ("identity_attack", "Does this comment attack someone based on their identity?"),
        ])
        yield noul_example(r["text"], question, float(r[field]), "civil_comments")


BUILDERS: dict[str, tuple[Callable[[random.Random, int], Iterator[Example]], int]] = {
    "boolq": (boolq, 3000),
    "mnli": (mnli, 3000),
    "stsb": (stsb, 2500),
    "emotion": (emotion, 2500),
    "ag_news": (ag_news, 2500),
    "dbpedia": (dbpedia, 2000),
    "banking77": (banking77, 3000),
    "massive": (massive_intent, 2500),
    "yelp": (yelp, 3000),
    "civil_comments": (civil_comments, 3000),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Jet training examples from public datasets.")
    ap.add_argument("--out", default="data/public.jsonl")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every per-source count")
    ap.add_argument("--only", nargs="*", help="subset of sources")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, (builder, count) in BUILDERS.items():
            if args.only and name not in args.only:
                continue
            n = 0
            for ex in builder(rng, int(count * args.scale)):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
            print(f"{name:>15}: {n}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
