"""The Metric protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task


class MetricNotApplicable(Exception):
    """Raised by a metric when it cannot score a given task.

    Example: a golden-set metric on a task that has no ``expected`` answer.
    :class:`~skill_factory.metrics.composite.CompositeMetric` catches this and
    drops the metric from that task's weighted average instead of penalizing it.
    """


@runtime_checkable
class Metric(Protocol):
    """Scores a single rollout in [0, 1] and returns actionable textual feedback.

    The feedback string is not cosmetic — reflective optimizers read it to decide
    how to edit the skill, so metrics should explain *why* a score was assigned.
    """

    name: str

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        ...
