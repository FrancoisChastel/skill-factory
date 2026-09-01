"""Golden-set metric: compare a rollout's output to a labeled expected answer."""

from __future__ import annotations

import json
import re
from enum import Enum

from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.metrics.base import MetricNotApplicable

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class MatchMode(str, Enum):
    """How an output is compared to its gold answer."""

    EXACT = "exact"
    NORMALIZED = "normalized"  # casefold + whitespace-collapse, then exact
    TOKEN_F1 = "token_f1"  # SQuAD-style token overlap (continuous)
    JSON_EQUAL = "json_equal"  # structural JSON compare with per-key partial credit
    NUMERIC = "numeric"  # first number within tolerance


class GoldenMetric:
    """Score output against ``task.expected``.

    Raises :class:`MetricNotApplicable` for tasks with no expected answer so a
    composite can drop it rather than penalize the skill.
    """

    def __init__(
        self,
        mode: MatchMode | str = MatchMode.NORMALIZED,
        *,
        name: str = "golden",
        numeric_tolerance: float = 1e-6,
    ):
        self.mode = MatchMode(mode)
        self.name = name
        self.numeric_tolerance = numeric_tolerance

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        if task.expected is None:
            raise MetricNotApplicable(f"task {task.id} has no expected answer")
        if not rollout.ok:
            return MetricResult(0.0, f"rollout failed: {rollout.error}", name=self.name)

        got = rollout.output
        want = task.expected

        if self.mode is MatchMode.EXACT:
            score = 1.0 if got.strip() == want.strip() else 0.0
        elif self.mode is MatchMode.NORMALIZED:
            score = 1.0 if _norm(got) == _norm(want) else 0.0
        elif self.mode is MatchMode.TOKEN_F1:
            score = _token_f1(got, want)
        elif self.mode is MatchMode.JSON_EQUAL:
            return self._json_result(got, want)
        elif self.mode is MatchMode.NUMERIC:
            score = self._numeric_score(got, want)
        else:  # pragma: no cover - exhaustive enum
            raise ValueError(f"unknown match mode: {self.mode}")

        verdict = "exact match" if score == 1.0 else f"partial/mismatch (score {score:.2f})"
        feedback = f"[{self.mode.value}] {verdict}. expected: {want!r}; got: {got.strip()!r}"
        return MetricResult(score, feedback, name=self.name)

    def _numeric_score(self, got: str, want: str) -> float:
        g, w = _first_number(got), _first_number(want)
        if g is None or w is None:
            return 0.0
        return 1.0 if abs(g - w) <= self.numeric_tolerance else 0.0

    def _json_result(self, got: str, want: str) -> MetricResult:
        got_obj, got_err = _parse_json(got)
        if got_err is not None:
            return MetricResult(
                0.0, f"output is not valid JSON: {got_err}", name=self.name
            )
        want_obj, _ = _parse_json(want)
        if isinstance(got_obj, dict) and isinstance(want_obj, dict):
            score, missing, wrong = _dict_key_f1(got_obj, want_obj)
            fb = f"[json_equal] key-F1 {score:.2f}."
            if missing:
                fb += f" missing/empty keys: {sorted(missing)}."
            if wrong:
                fb += f" wrong values: {sorted(wrong)}."
            return MetricResult(score, fb, breakdown={"key_f1": score}, name=self.name)
        score = 1.0 if got_obj == want_obj else 0.0
        return MetricResult(
            score, f"[json_equal] structural {'match' if score else 'mismatch'}", name=self.name
        )


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().casefold())


def _tokens(text: str) -> list[str]:
    return _norm(text).split()


def _token_f1(got: str, want: str) -> float:
    g, w = _tokens(got), _tokens(want)
    if not g and not w:
        return 1.0
    if not g or not w:
        return 0.0
    common: dict[str, int] = {}
    w_counts: dict[str, int] = {}
    for t in w:
        w_counts[t] = w_counts.get(t, 0) + 1
    overlap = 0
    for t in g:
        if w_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(g)
    recall = overlap / len(w)
    return 2 * precision * recall / (precision + recall)


def _first_number(text: str) -> float | None:
    m = _NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


def _parse_json(text: str) -> tuple[object, str | None]:
    """Parse JSON, tolerating fenced code blocks and surrounding prose."""
    candidate = _strip_code_fence(text).strip()
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first balanced {...} or [...] span.
    span = _first_json_span(candidate)
    if span is not None:
        try:
            return json.loads(span), None
        except json.JSONDecodeError as exc:
            return None, str(exc)
    return None, "no JSON object found"


def _strip_code_fence(text: str) -> str:
    fence = re.match(r"\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return fence.group(1) if fence else text


def _first_json_span(text: str) -> str | None:
    starts = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in starts:
            depth = 0
            for j in range(i, len(text)):
                if text[j] in starts:
                    depth += 1
                elif text[j] in starts.values():
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            return None
    return None


def _dict_key_f1(got: dict, want: dict) -> tuple[float, set[str], set[str]]:
    """Per-key F1: reward matching keys with equal (stringified) values."""
    want_keys = set(want)
    correct = set()
    wrong = set()
    for k in want_keys:
        if k in got and _values_equal(got[k], want[k]):
            correct.add(k)
        elif k in got:
            wrong.add(k)
    missing = want_keys - set(got)
    predicted = set(got)
    if not want_keys and not predicted:
        return 1.0, set(), set()
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(want_keys) if want_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, missing, wrong


def _values_equal(a: object, b: object) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return _norm(a) == _norm(b)
    return a == b
