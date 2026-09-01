"""The Rollout primitive: the result of running a skill on a task in some harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from skill_factory.core.task import Task


@dataclass(frozen=True)
class Rollout:
    """A single execution of a skill against a task.

    Attributes:
        task: The task that was run.
        output: The skill's final textual output (empty string on failure).
        trajectory: Optional intermediate steps (tool calls, reasoning traces).
        error: Populated if the harness failed to produce a rollout.
        metadata: Free-form execution info (latency, tokens, model, ...).
    """

    task: Task
    output: str
    trajectory: Sequence[Any] = field(default_factory=tuple)
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if the rollout produced output without a harness-level error."""
        return self.error is None

    @classmethod
    def failed(cls, task: Task, error: str) -> "Rollout":
        """Construct a failed rollout (skill/harness error) with empty output."""
        return cls(task=task, output="", error=error)
