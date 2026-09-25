"""Prompt format shared by training and inference.

Every question is reduced to "pick one label token". The model is run once over
the prompt and we read the next-token logits restricted to the label tokens:

- choice: one label per option (A..Z, then single-token AA, AB, ...), up to 255 options
- score:  one digit per level (0..9), 2-10 levels, ordered low -> high
- noul:   "no" / "yes"
"""

from __future__ import annotations

import itertools
import json
import string
from dataclasses import dataclass
from typing import Any, Literal

QuestionType = Literal["choice", "score", "noul"]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10

SYSTEM_PROMPT = (
    "You are Jet, a decision model. Read the state and the question, then answer "
    "with exactly one label from the allowed labels."
)


@dataclass(frozen=True)
class Question:
    type: QuestionType
    instructions: str
    # choice: {key: description}; score: [level descriptions, low -> high]; noul: None
    criteria: dict[str, str] | list[str] | None = None

    @property
    def keys(self) -> list[str]:
        """Answer keys in label order (option keys, level descriptions, or no/yes)."""
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return list(self.criteria)
        return ["no", "yes"]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        q = cls(type=d["type"], instructions=d["instructions"], criteria=d.get("criteria"))
        q.validate()
        return q

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d

    def validate(self) -> None:
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type {self.type!r}")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be a non-empty string")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice criteria must be an object with 2-{MAX_CHOICE_OPTIONS} options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
                raise ValueError(f"score criteria must be a list of {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels")
        elif self.criteria is not None:
            raise ValueError("noul questions take no criteria")


_CHOICE_LABELS: dict[int, list[str]] = {}


def choice_labels(tokenizer, n: int) -> list[str]:
    """A..Z, then the two-letter labels AA, AB, ... that are single tokens for this tokenizer."""
    key = id(tokenizer)
    if key not in _CHOICE_LABELS:
        letters = string.ascii_uppercase
        pairs = ("".join(p) for p in itertools.product(letters, repeat=2))
        candidates = list(letters) + [p for p in pairs if len(tokenizer.encode(p, add_special_tokens=False)) == 1]
        if len(candidates) < MAX_CHOICE_OPTIONS:
            raise ValueError(f"tokenizer only has {len(candidates)} single-token choice labels")
        _CHOICE_LABELS[key] = candidates[:MAX_CHOICE_OPTIONS]
    return _CHOICE_LABELS[key][:n]


def labels_for(tokenizer, q: Question) -> list[str]:
    if q.type == "choice":
        return choice_labels(tokenizer, len(q.criteria))
    if q.type == "score":
        return [str(i) for i in range(len(q.criteria))]
    return ["no", "yes"]


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, list) and all(isinstance(s, str) for s in state):
        return "\n\n".join(state)
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_user(state: Any, q: Question, labels: list[str]) -> str:
    parts = [f"<state>\n{render_state(state)}\n</state>", ""]
    if q.type == "choice":
        opts = "\n".join(f"{lab}: {key}: {desc}" for lab, (key, desc) in zip(labels, q.criteria.items()))
        parts += [
            f"Question: {q.instructions}",
            f"Options:\n{opts}",
            f"Answer with the label of the best option ({labels[0]}-{labels[-1]}).",
        ]
    elif q.type == "score":
        levels = "\n".join(f"{lab}: {desc}" for lab, desc in zip(labels, q.criteria))
        parts += [
            f"Question: {q.instructions}",
            f"Scale (lowest to highest):\n{levels}",
            f"Answer with the level number (0-{len(labels) - 1}).",
        ]
    else:
        parts += [f"Question: {q.instructions}", "Answer yes or no."]
    return "\n".join(parts)


def build_prompt(tokenizer, state: Any, q: Question) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_user(state, q, labels_for(tokenizer, q))}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def _label_token_id(tokenizer, label: str) -> int:
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"label {label!r} is not a single token for this tokenizer")
    return ids[0]


def label_token_ids(tokenizer, q: Question) -> list[int]:
    return [_label_token_id(tokenizer, lab) for lab in labels_for(tokenizer, q)]
