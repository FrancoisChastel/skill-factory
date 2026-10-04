"""System One metric: yes/no rubric checks answered by a decision model (jev).

An LLM-as-judge writes prose and is slow and expensive per rollout. A decision
model answers typed questions with calibrated probabilities in a few hundred
milliseconds, and bills the state once for every question asked about it. For
rubric checks like "is the output strict JSON with no prose?" or "does the
summary invent a figure the input lacks?", that makes each optimization round
much cheaper:

    metric:
      systemone:
        weight: 1.0
        provider: typesafe          # see skill_factory.classifier.systemone.PROVIDERS
        model: jev-latest
        checks:
          - name: strict_json
            question: Is the OUTPUT strict JSON, with no prose or code fences around it?
            true: "Yes: the output is strict JSON and nothing else."
            false: "No: the output has prose, fences, or is not valid JSON."
          - name: no_invention
            question: Does every value in the OUTPUT appear in, or follow from, the INPUT?
            true: "Yes: nothing in the output is invented."
            false: "No: the output contains values the input does not support."
            weight: 2

The score is the weighted mean of P(true); with ``threshold`` each check counts
1 at or above it and 0 below. Answers are cached by (state, question, model), so
a rollout seen before costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.questions import Question, sha24
from skill_factory.classifier.systemone import SystemOneClient, SystemOneError
from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task

_INSTRUCTIONS = (
    "You are checking one output of an AI agent against its input. The state holds the INPUT the agent "
    "received and the OUTPUT it produced. Answer only about the output.\n\n"
)


@dataclass(frozen=True)
class Check:
    name: str
    question: str
    true: str
    false: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Check.weight must be positive")

    def compiled(self) -> Question:
        return Question(_INSTRUCTIONS + self.question, self.true, self.false)


class SystemOneMetric:
    """Score a rollout by asking a System One model yes/no checks about it."""

    def __init__(
        self,
        client: SystemOneClient,
        checks: Sequence[Check],
        *,
        name: str = "systemone",
        threshold: float | None = None,
        include_expected: bool = False,
        cache: AnswerCache | None = None,
    ):
        if not checks:
            raise ValueError("SystemOneMetric needs at least one check")
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise ValueError("check names must be unique")
        self._client = client
        self._checks = list(checks)
        self.name = name
        self._threshold = threshold
        self._include_expected = include_expected
        self._cache = cache if cache is not None else AnswerCache(None)

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        if not rollout.ok:
            return MetricResult(0.0, f"rollout failed: {rollout.error}", name=self.name)
        state = self._state(task, rollout)
        try:
            probabilities = self._answers(state)
        except SystemOneError as exc:
            return MetricResult(0.0, f"System One check failed: {exc}", name=self.name)
        breakdown, total, weights = {}, 0.0, 0.0
        for c in self._checks:
            p = probabilities[c.name]
            value = (1.0 if p >= self._threshold else 0.0) if self._threshold is not None else p
            breakdown[c.name] = value
            total += value * c.weight
            weights += c.weight
        failing = [c for c in self._checks if probabilities[c.name] < 0.5]
        feedback = "; ".join(f"{c.name}: P(true)={probabilities[c.name]:.2f} ({c.false})" for c in failing)
        return MetricResult(
            total / weights,
            feedback or "all checks pass",
            breakdown=breakdown,
            name=self.name,
        )

    def _state(self, task: Task, rollout: Rollout) -> str:
        parts = ["=== INPUT ===", task.input, "", "=== OUTPUT ===", rollout.output or "(empty)"]
        if self._include_expected and task.expected is not None:
            parts += ["", "=== REFERENCE ===", task.expected]
        return "\n".join(parts)

    def _answers(self, state: str) -> dict[str, float]:
        s = sha24(state)
        model = self._client.model
        questions = {c.name: c.compiled() for c in self._checks}
        cached = {n: self._cache.get(s, q.hash, model) for n, q in questions.items()}
        missing = {n: questions[n] for n, p in cached.items() if p is None}
        if missing:
            answers = self._client.ask(state, missing)
            self._cache.put_many(
                {"s": s, "q": questions[n].hash, "m": model, "p": answers.probabilities[n], "v": answers.model}
                for n in missing
            )
            cached.update(answers.probabilities)
        return {n: float(p) for n, p in cached.items() if p is not None}
