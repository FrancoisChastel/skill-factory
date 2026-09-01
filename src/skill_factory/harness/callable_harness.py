"""A harness backed by an arbitrary Python callable.

Useful for (a) deterministic tests, and (b) skills whose "execution" is a local
function rather than an LLM call (e.g. simulating a skill's effect on inputs).
"""

from __future__ import annotations

from typing import Callable

from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task


class CallableHarness:
    """Wrap ``fn(skill, task) -> str`` as a Harness.

    The callable receives the full Skill (so it can read the body being optimized)
    and the Task. Any exception is captured as a failed rollout.
    """

    def __init__(self, fn: Callable[[Skill, Task], str]):
        self._fn = fn

    def run(self, skill: Skill, task: Task) -> Rollout:
        try:
            output = self._fn(skill, task)
        except Exception as exc:  # noqa: BLE001 - never abort the epoch on one task
            return Rollout.failed(task, f"{type(exc).__name__}: {exc}")
        return Rollout(task=task, output=output)
