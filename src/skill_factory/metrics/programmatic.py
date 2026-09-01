"""Programmatic metric: deterministic, objective checks on rollout output.

A check is ``fn(task, rollout) -> (score in [0,1], feedback)``. Compose several;
the metric reports their mean and concatenates feedback. The :data:`checks`
namespace provides ready-made builders.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.metrics.golden import _parse_json  # reuse tolerant JSON parsing

Check = Callable[[Task, Rollout], "tuple[float, str]"]


class ProgrammaticMetric:
    """Run one or more deterministic checks and average their scores."""

    def __init__(self, check_fns: Sequence[Check], *, name: str = "programmatic"):
        if not check_fns:
            raise ValueError("ProgrammaticMetric needs at least one check")
        self._checks = list(check_fns)
        self.name = name

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        if not rollout.ok:
            return MetricResult(0.0, f"rollout failed: {rollout.error}", name=self.name)
        scores: list[float] = []
        feedback: list[str] = []
        breakdown: dict[str, float] = {}
        for i, check in enumerate(self._checks):
            score, note = check(task, rollout)
            score = max(0.0, min(1.0, float(score)))
            scores.append(score)
            label = getattr(check, "check_name", f"check_{i}")
            breakdown[label] = score
            feedback.append(f"[{label}] {'pass' if score == 1.0 else f'{score:.2f}'}: {note}")
        mean_score = sum(scores) / len(scores)
        return MetricResult(mean_score, "\n".join(feedback), breakdown=breakdown, name=self.name)


def _named(fn: Check, name: str) -> Check:
    fn.check_name = name  # type: ignore[attr-defined]
    return fn


class checks:
    """Factory namespace of common checks (call these to build a Check)."""

    @staticmethod
    def is_valid_json() -> Check:
        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            _, err = _parse_json(rollout.output)
            return (1.0, "valid JSON") if err is None else (0.0, f"invalid JSON: {err}")

        return _named(_check, "is_valid_json")

    @staticmethod
    def json_has_keys(*keys: str) -> Check:
        required = list(keys)

        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            obj, err = _parse_json(rollout.output)
            if err is not None or not isinstance(obj, dict):
                return 0.0, f"not a JSON object: {err or type(obj).__name__}"
            present = [k for k in required if k in obj]
            score = len(present) / len(required) if required else 1.0
            missing = [k for k in required if k not in obj]
            return score, (f"missing keys: {missing}" if missing else "all keys present")

        return _named(_check, "json_has_keys")

    @staticmethod
    def matches_regex(pattern: str, *, flags: int = 0) -> Check:
        compiled = re.compile(pattern, flags)

        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            ok = compiled.search(rollout.output) is not None
            return (1.0, f"matched /{pattern}/") if ok else (0.0, f"did not match /{pattern}/")

        return _named(_check, "matches_regex")

    @staticmethod
    def contains(substring: str, *, case_sensitive: bool = False) -> Check:
        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            hay = rollout.output if case_sensitive else rollout.output.lower()
            needle = substring if case_sensitive else substring.lower()
            ok = needle in hay
            return (1.0, f"contains {substring!r}") if ok else (0.0, f"missing {substring!r}")

        return _named(_check, "contains")

    @staticmethod
    def not_contains(substring: str, *, case_sensitive: bool = False) -> Check:
        base = checks.contains(substring, case_sensitive=case_sensitive)

        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            score, note = base(task, rollout)
            return (1.0 - score), ("must not contain " + note)

        return _named(_check, "not_contains")

    @staticmethod
    def max_length(max_chars: int) -> Check:
        def _check(task: Task, rollout: Rollout) -> tuple[float, str]:
            n = len(rollout.output)
            ok = n <= max_chars
            return (1.0, f"{n} <= {max_chars} chars") if ok else (0.0, f"{n} > {max_chars} chars")

        return _named(_check, "max_length")

    @staticmethod
    def custom(fn: Check, *, name: str = "custom") -> Check:
        """Wrap a user-provided ``fn(task, rollout) -> (score, feedback)``."""
        return _named(fn, name)


__all__ = ["ProgrammaticMetric", "checks", "Check"]
