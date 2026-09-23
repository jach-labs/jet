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


# Score questions with varied scales. The model must read the levels rather than learn one fixed
# scale per dataset: the number of levels changes, the wording changes, and some questions ask
# the reverse direction ("how different", "how negative") so the target flips.

STSB_SCALES = {
    3: ["unrelated", "related, but the meaning differs", "same meaning"],
    4: ["nothing in common", "same topic, different meaning", "similar meaning, some details differ", "identical meaning"],
    5: ["not similar at all", "slightly similar", "somewhat similar", "very similar", "they say the same thing"],
    6: ["completely unrelated", "different topics but share a few details", "not equivalent but on the same topic",
        "roughly equivalent, some important details differ", "mostly equivalent, only minor details differ",
        "completely equivalent in meaning"],
}
STSB_Q = ["How similar in meaning are these two sentences?", "Do the two sentences say the same thing?",
          "Rate the semantic overlap between sentence_1 and sentence_2."]
STSB_REVERSED = {
    3: ["same meaning", "related, but the meaning differs", "unrelated"],
    5: ["no difference at all", "minor differences", "noticeably different", "mostly different", "completely different"],
}
STSB_REVERSED_Q = ["How different in meaning are the two sentences?", "How far apart are these sentences in meaning?"]


def rescale(value: float, top: float, k: int) -> float:
    """Map a value on 0..top onto level indices 0..k-1."""
    return min(value / top * (k - 1), k - 1)


def stsb_scales(rng: random.Random, n: int) -> Iterator[Example]:
    for r in rows("nyu-mll/glue", "stsb", "train", n, rng.randint(0, 10**6)):
        state = {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}
        if rng.random() < 0.25:
            k = rng.choice(list(STSB_REVERSED))
            yield score_example(state, rng.choice(STSB_REVERSED_Q), STSB_REVERSED[k], rescale(5.0 - r["label"], 5.0, k), "stsb")
        else:
            k = rng.choice(list(STSB_SCALES))
            yield score_example(state, rng.choice(STSB_Q), STSB_SCALES[k], rescale(r["label"], 5.0, k), "stsb")


YELP_SCALES = [
    ("How positive is this review?", ["negative", "mixed", "positive"], False),
    ("How satisfied is the customer?", ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"], False),
    ("How many stars did the reviewer give?", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], False),
    ("Would this customer recommend the business?", ["definitely not", "probably not", "maybe", "probably", "definitely"], False),
    ("How negative is this review?", ["not negative", "somewhat negative", "very negative"], True),
    ("How angry or disappointed is the customer?", ["not at all", "a little", "moderately", "very", "extremely"], True),
]


def yelp_scales(rng: random.Random, n: int) -> Iterator[Example]:
    for r in rows("Yelp/yelp_review_full", None, "train", n, rng.randint(0, 10**6)):
        instructions, levels, reverse = rng.choice(YELP_SCALES)
        stars = 4.0 - r["label"] if reverse else float(r["label"])
        yield score_example(r["text"], instructions, levels, rescale(stars, 4.0, len(levels)), "yelp")


CIVIL_SCALES = [
    ("How toxic is this comment?", ["not toxic", "somewhat toxic", "very toxic"]),
    ("How rude or disrespectful is this comment?", ["polite", "slightly rude", "rude", "very rude", "extremely rude"]),
    ("How acceptable is this comment for a family-friendly forum?", ["fine", "borderline", "unacceptable"]),
    ("How likely would moderators remove this comment?", ["very unlikely", "unlikely", "maybe", "likely", "very likely"]),
]


def civil_scores(rng: random.Random, n: int) -> Iterator[Example]:
    # toxicity is the fraction of raters who marked the comment toxic; read as a 0-1 intensity.
    ds = load_dataset("google/civil_comments", split="train").shuffle(seed=rng.randint(0, 10**6)).select(range(n * 8))
    toxic = [r for r in ds if r["toxicity"] >= 0.3][: n // 2]
    clean = [r for r in ds if r["toxicity"] < 0.3][: n - len(toxic)]
    for r in toxic + clean:
        instructions, levels = rng.choice(CIVIL_SCALES)
        yield score_example(r["text"], instructions, levels, rescale(r["toxicity"], 1.0, len(levels)), "civil_comments")


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
    "stsb_scales": (stsb_scales, 2000),
    "yelp_scales": (yelp_scales, 2000),
    "civil_scores": (civil_scores, 1500),
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


def score_eval_main() -> None:
    """Score questions from official held-out splits (public.py builds from train splits only).

    stsb and yelp measure the trained score tasks on far more rows than the test split holds;
    sst5 and amazon are ordinal tasks the model never trains on.
    """
    ap = argparse.ArgumentParser(description="Build an ordinal (score) eval set from held-out splits.")
    ap.add_argument("--out", default="data/score_eval.jsonl")
    ap.add_argument("--n", type=int, default=600, help="rows per source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stars = ["very negative", "negative", "mixed or neutral", "positive", "very positive"]
    stsb_levels = [
        "completely unrelated",
        "different topics but share a few details",
        "not equivalent but on the same topic",
        "roughly equivalent, some important details differ",
        "mostly equivalent, only minor details differ",
        "completely equivalent in meaning",
    ]
    sources = {
        "stsb": (rows("nyu-mll/glue", "stsb", "validation", args.n, args.seed),
                 lambda r: ({"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, float(r["label"])),
                 "How similar in meaning are the two sentences?", stsb_levels),
        "yelp": (rows("Yelp/yelp_review_full", None, "test", args.n, args.seed),
                 lambda r: (r["text"], float(r["label"])), "How positive is this review?", stars),
        "sst5": (rows("SetFit/sst5", None, "test", args.n, args.seed),
                 lambda r: (r["text"], float(r["label"])), "What does the critic think of the movie?",
                 ["hated it", "disliked it", "neither liked nor disliked it", "liked it", "loved it"]),
        "amazon": (rows("SetFit/amazon_reviews_multi_en", None, "test", args.n, args.seed),
                   lambda r: (r["text"], float(r["label"])), "How many stars would this buyer give the product?",
                   ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, (data, parse, instructions, levels) in sources.items():
            for r in data:
                state, value = parse(r)
                f.write(json.dumps(score_example(state, instructions, levels, value, name), ensure_ascii=False) + "\n")
            print(f"{name:>8}: {len(data)}")
    print(f"wrote {out}")
