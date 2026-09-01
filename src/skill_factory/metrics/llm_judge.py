"""LLM-as-judge metric: rubric-based scoring for open-ended skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.llm.client import LLMClient

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluation judge. You score how well an ASSISTANT "
    "OUTPUT satisfies a TASK, using the provided rubric. Reward correctness and "
    "completeness; penalize hallucination, format violations, and omissions. "
    "Return ONLY minified JSON — no prose, no code fences."
)


@dataclass(frozen=True)
class RubricCriterion:
    """One scored dimension of the rubric.

    Attributes:
        name: Short identifier (e.g. "correctness").
        description: What the judge should look for.
        weight: Relative importance within the rubric.
    """

    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("RubricCriterion.weight must be positive")


class LLMJudgeMetric:
    """Score a rollout against a weighted rubric using a judge LLM.

    The judge returns a per-criterion score in [0, 10] plus a rationale. Scores
    are normalized to [0, 1] and combined by criterion weight. The rationale is
    surfaced as feedback so the optimizer can act on it.
    """

    def __init__(
        self,
        client: LLMClient,
        criteria: Sequence[RubricCriterion],
        *,
        name: str = "llm_judge",
        temperature: float = 0.0,
        include_expected: bool = True,
    ):
        if not criteria:
            raise ValueError("LLMJudgeMetric needs at least one rubric criterion")
        self._client = client
        self._criteria = list(criteria)
        self.name = name
        self._temperature = temperature
        self._include_expected = include_expected

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        if not rollout.ok:
            return MetricResult(0.0, f"rollout failed: {rollout.error}", name=self.name)

        prompt = self._build_prompt(task, rollout)
        raw = self._client.generate(
            prompt, system=_JUDGE_SYSTEM, temperature=self._temperature, max_tokens=1024
        )
        parsed = _parse_judge_json(raw)
        if parsed is None:
            return MetricResult(
                0.0, f"judge returned unparseable output: {raw[:200]!r}", name=self.name
            )
        return self._aggregate(parsed)

    def _build_prompt(self, task: Task, rollout: Rollout) -> str:
        rubric_lines = [
            f"- {c.name} (weight {c.weight}): {c.description}" for c in self._criteria
        ]
        schema_keys = ", ".join(f'"{c.name}": <0-10>' for c in self._criteria)
        parts = [
            "TASK INPUT:",
            task.input,
            "",
            "ASSISTANT OUTPUT:",
            rollout.output or "(empty)",
        ]
        if self._include_expected and task.expected is not None:
            parts += ["", "REFERENCE ANSWER (for grading only):", task.expected]
        parts += [
            "",
            "RUBRIC:",
            *rubric_lines,
            "",
            "Score each criterion from 0 (fails) to 10 (perfect). Respond with JSON:",
            f'{{"scores": {{{schema_keys}}}, "rationale": "<one paragraph>"}}',
        ]
        return "\n".join(parts)

    def _aggregate(self, parsed: dict) -> MetricResult:
        scores = parsed.get("scores", {})
        rationale = str(parsed.get("rationale", "")).strip()
        total_weight = 0.0
        weighted = 0.0
        breakdown: dict[str, float] = {}
        for c in self._criteria:
            raw_score = _clamp_score(scores.get(c.name))
            normalized = raw_score / 10.0
            breakdown[c.name] = normalized
            weighted += normalized * c.weight
            total_weight += c.weight
        final = weighted / total_weight if total_weight else 0.0
        detail = ", ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
        feedback = f"{rationale} (per-criterion: {detail})".strip()
        return MetricResult(final, feedback, breakdown=breakdown, name=self.name)


def _clamp_score(value: object) -> float:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, num))


def _parse_judge_json(raw: str) -> dict | None:
    text = raw.strip()
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
