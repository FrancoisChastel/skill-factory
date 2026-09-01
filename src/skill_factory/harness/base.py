"""The Harness protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task


@runtime_checkable
class Harness(Protocol):
    """Runs a skill on a task, returning a Rollout.

    Implementations must never raise for a per-task failure; instead they return
    ``Rollout.failed(task, error)`` so a single bad rollout cannot abort a whole
    optimization epoch.
    """

    def run(self, skill: Skill, task: Task) -> Rollout:
        ...
