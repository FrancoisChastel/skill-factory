"""Composite metric: weighted combination of golden / programmatic / judge metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from skill_factory.core.result import MetricResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.metrics.base import Metric, MetricNotApplicable


@dataclass(frozen=True)
class WeightedMetric:
    """A metric paired with its weight in a composite."""

    metric: Metric
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("WeightedMetric.weight must be positive")


class CompositeMetric:
    """Combine several metrics into one weighted score with merged feedback.

    Metrics that raise :class:`MetricNotApplicable` for a task (e.g. a golden
    metric on an unlabeled task) are dropped from that task's average rather than
    scored zero, so mixing labeled and unlabeled tasks is safe.
    """

    def __init__(self, components: Sequence[WeightedMetric], *, name: str = "composite"):
        if not components:
            raise ValueError("CompositeMetric needs at least one component")
        self._components = list(components)
        self.name = name

    @classmethod
    def of(cls, *metrics: Metric, name: str = "composite") -> "CompositeMetric":
        """Build an equally-weighted composite from bare metrics."""
        return cls([WeightedMetric(m) for m in metrics], name=name)

    def evaluate(self, task: Task, rollout: Rollout) -> MetricResult:
        total_weight = 0.0
        weighted_sum = 0.0
        feedback: list[str] = []
        breakdown: dict[str, float] = {}
        applied = 0

        for component in self._components:
            metric = component.metric
            metric_name = getattr(metric, "name", metric.__class__.__name__)
            try:
                result = metric.evaluate(task, rollout)
            except MetricNotApplicable as exc:
                feedback.append(f"[{metric_name}] skipped: {exc}")
                continue
            applied += 1
            weighted_sum += result.score * component.weight
            total_weight += component.weight
            breakdown[metric_name] = result.score
            feedback.append(f"[{metric_name}] score {result.score:.2f}: {result.feedback}")

        if applied == 0:
            return MetricResult(
                0.0, "no metric was applicable to this task", name=self.name
            )
        final = weighted_sum / total_weight
        return MetricResult(final, "\n".join(feedback), breakdown=breakdown, name=self.name)
