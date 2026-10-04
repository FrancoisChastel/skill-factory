"""The probe set: the trainable artifact of classifier mode.

Where skill mode trains one SKILL.md text, classifier mode trains a probe set:

    questions    text parameters (each a System One choice question, maybe framed)
    score        how answers combine into one number (a small expression tree)
    threshold    at or above it, an example is flagged (null: fit it on train)
    state_budget characters of state sent per example; longer states are skipped, never truncated
    model        the model the threshold was measured on (pinned, see drift.py)

Score expressions are JSON so they can be stored, hashed, diffed and exported to
production code byte for byte:

    "qid"                          P(true) of a question
    0.5                            a constant
    {"mean": [e, ...]}             also "max", "min", "sum"
    {"weighted": [[w, e], ...]}    sum of w * e
    {"scale": [e, k]}              e * k

jev's shipped score, half the mean of five intent questions plus half a lure:
``{"mean": [{"mean": [q1, q2, q3, q4, q5]}, "lure"]}`` at threshold 0.075.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Union

from skill_factory.classifier.questions import QuestionBank

Score = Union[str, float, int, dict]

_REDUCERS = ("mean", "max", "min", "sum")


def referenced_ids(expr: Score) -> list[str]:
    """Question ids an expression reads, in first-use order."""
    seen: list[str] = []

    def walk(e: Score) -> None:
        if isinstance(e, str):
            if e not in seen:
                seen.append(e)
        elif isinstance(e, dict):
            op, arg = _single(e)
            if op in _REDUCERS:
                for x in arg:
                    walk(x)
            elif op == "weighted":
                for _, x in arg:
                    walk(x)
            elif op == "scale":
                walk(arg[0])

    walk(expr)
    return seen


def validate_score(expr: Score) -> None:
    """Raise ValueError on a malformed expression (before it is stored or exported)."""
    if isinstance(expr, bool):
        raise ValueError("a score cannot be a boolean")
    if isinstance(expr, (str, int, float)):
        return
    if not isinstance(expr, dict):
        raise ValueError(f"invalid score expression: {expr!r}")
    op, arg = _single(expr)
    if op in _REDUCERS:
        if not isinstance(arg, list) or not arg:
            raise ValueError(f"'{op}' needs a non-empty list")
        for x in arg:
            validate_score(x)
    elif op == "weighted":
        if not isinstance(arg, list) or not arg:
            raise ValueError("'weighted' needs a non-empty list of [weight, expression]")
        for pair in arg:
            if not (isinstance(pair, list) and len(pair) == 2 and _is_number(pair[0])):
                raise ValueError(f"'weighted' entries are [weight, expression], got {pair!r}")
            validate_score(pair[1])
    elif op == "scale":
        if not (isinstance(arg, list) and len(arg) == 2 and _is_number(arg[1])):
            raise ValueError("'scale' needs [expression, factor]")
        validate_score(arg[0])
    else:
        raise ValueError(f"unknown score operator {op!r}")


def evaluate_score(expr: Score, p: Mapping[str, float]) -> float | None:
    """The score for one example's answers; None when a question it reads was not answered."""
    if isinstance(expr, str):
        value = p.get(expr)
        return None if value is None else float(value)
    if isinstance(expr, (int, float)):
        return float(expr)
    op, arg = _single(expr)
    if op in _REDUCERS:
        values = [evaluate_score(x, p) for x in arg]
        if any(v is None for v in values):
            return None
        nums = [float(v) for v in values if v is not None]
        if op == "mean":
            return sum(nums) / len(nums)
        if op == "sum":
            return sum(nums)
        return max(nums) if op == "max" else min(nums)
    if op == "weighted":
        total = 0.0
        for w, x in arg:
            v = evaluate_score(x, p)
            if v is None:
                return None
            total += float(w) * v
        return total
    if op == "scale":
        v = evaluate_score(arg[0], p)
        return None if v is None else v * float(arg[1])
    raise ValueError(f"unknown score operator {op!r}")


def describe_score(expr: Score) -> str:
    """A short human-readable form, e.g. ``mean(mean(a, b), c)``."""
    if isinstance(expr, str):
        return expr
    if isinstance(expr, (int, float)):
        return f"{expr:g}"
    op, arg = _single(expr)
    if op in _REDUCERS:
        return f"{op}({', '.join(describe_score(x) for x in arg)})"
    if op == "weighted":
        return " + ".join(f"{float(w):g}*{describe_score(x)}" for w, x in arg)
    return f"{describe_score(arg[0])}*{float(arg[1]):.4g}"


@dataclass(frozen=True)
class ProbeSet:
    """Questions, a score over their answers, a threshold, and the pipeline settings they were measured with.

    ``bank`` may hold questions the score does not read: a policy can use them
    (jev's capability probes only doubt static findings; they never add one).
    """

    name: str
    bank: QuestionBank
    score: Score
    threshold: float | None
    model: str = "jev-latest"
    pinned_version: str | None = None
    state_budget: int = 96_000
    calibration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_score(self.score)
        unknown = [q for q in referenced_ids(self.score) if q not in self.bank]
        if unknown:
            raise ValueError(f"score reads questions not in the probe set: {unknown}")
        if self.state_budget < 1:
            raise ValueError("state_budget must be >= 1")

    @property
    def score_ids(self) -> list[str]:
        return referenced_ids(self.score)

    def score_of(self, p: Mapping[str, float]) -> float | None:
        return evaluate_score(self.score, p)

    def flags(self, p: Mapping[str, float]) -> bool:
        s = self.score_of(p)
        return s is not None and self.threshold is not None and s >= self.threshold

    def with_changes(self, **changes: Any) -> "ProbeSet":
        return replace(self, **changes)

    def content_hash(self) -> str:
        """SHA-256 of what decides a verdict: compiled questions, score, threshold, model and budget.

        The name and calibration notes are not part of it, so renaming a probe set
        does not break a parity check, and changing one word of a question does.
        """
        payload = {
            "questions": {
                qid: [q.instructions, q.true, q.false] for qid, q in sorted(self.bank.compiled().items())
            },
            "score": self.score,
            "threshold": self.threshold,
            "model": self.model,
            "state_budget": self.state_budget,
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "pinned_version": self.pinned_version,
            "state_budget": self.state_budget,
            **self.bank.to_dict(),
            "score": self.score,
            "threshold": self.threshold,
            "calibration": dict(self.calibration),
            "sha256": self.content_hash(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProbeSet":
        if not isinstance(data, Mapping):
            raise ValueError("a probe set must be a mapping")
        for key in ("questions", "score"):
            if key not in data:
                raise ValueError(f"probe set is missing {key!r}")
        threshold = data.get("threshold")
        probeset = cls(
            name=str(data.get("name", "probeset")),
            bank=QuestionBank.from_dict({"frames": data.get("frames") or {}, "questions": data["questions"]}),
            score=data["score"],
            threshold=float(threshold) if threshold is not None else None,
            model=str(data.get("model", "jev-latest")),
            pinned_version=data.get("pinned_version"),
            state_budget=int(data.get("state_budget", 96_000)),
            calibration=dict(data.get("calibration") or {}),
        )
        stated = data.get("sha256")
        if stated and stated != probeset.content_hash():
            raise ValueError(
                "probe set content does not match its sha256: it was edited after it was "
                "measured; remove the 'sha256' field to accept the edit"
            )
        return probeset

    @classmethod
    def load(cls, path: str | Path) -> "ProbeSet":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"probe set not found: {p}")
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}: invalid JSON: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return p


def _single(expr: dict) -> tuple[str, Any]:
    if len(expr) != 1:
        raise ValueError(f"a score operator has exactly one key, got {sorted(expr)}")
    ((op, arg),) = expr.items()
    return str(op), arg


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
