"""Result primitives: metric outputs, per-task evaluations, and optimization runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Mapping, Sequence

from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task


@dataclass(frozen=True)
class MetricResult:
    """The score and *textual feedback* a metric assigns to one rollout.

    Textual feedback is load-bearing: reflective optimizers (GEPA, SkillOpt, the
    built-in loop) read it to decide how to edit the skill. A bare number is not
    enough to improve a prompt.
    """

    score: float
    feedback: str = ""
    breakdown: Mapping[str, float] = field(default_factory=dict)
    name: str = "metric"

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"MetricResult.score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class TaskEvaluation:
    """A task, its rollout, and the metric result — the unit of an Evaluation."""

    task: Task
    rollout: Rollout
    result: MetricResult


@dataclass(frozen=True)
class Evaluation:
    """Aggregate result of scoring a skill across a dataset split."""

    items: Sequence[TaskEvaluation]

    @property
    def score(self) -> float:
        """Mean score across all tasks (0.0 if empty)."""
        if not self.items:
            return 0.0
        return mean(item.result.score for item in self.items)

    @property
    def size(self) -> int:
        return len(self.items)

    def worst(self, k: int = 3) -> list[TaskEvaluation]:
        """Return the ``k`` lowest-scoring evaluations — the optimizer's focus set."""
        return sorted(self.items, key=lambda i: i.result.score)[:k]

    def feedback_digest(self, k: int = 5) -> str:
        """Concatenate feedback from the worst ``k`` tasks for reflective editing."""
        lines = []
        for item in self.worst(k):
            lines.append(
                f"- task {item.task.id} (score {item.result.score:.2f}): "
                f"{item.result.feedback.strip() or 'no feedback'}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate skill produced during optimization, with its validation score."""

    iteration: int
    skill: Skill
    train_score: float
    val_score: float
    accepted: bool
    note: str = ""


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of an optimization run.

    Attributes:
        best_skill: The highest-scoring skill on the validation split.
        baseline_score: Validation score of the seed skill.
        best_score: Validation score of ``best_skill``.
        history: Every candidate considered, in order.
        optimizer: Name of the optimizer backend that produced this result.
        metadata: Backend-specific extras (log dirs, api call counts, ...).
    """

    best_skill: Skill
    baseline_score: float
    best_score: float
    history: Sequence[CandidateRecord]
    optimizer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Absolute validation-score gain over the seed skill."""
        return self.best_score - self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "baseline_score": round(self.baseline_score, 4),
            "best_score": round(self.best_score, 4),
            "improvement": round(self.improvement, 4),
            "candidates": len(self.history),
            "accepted": sum(1 for c in self.history if c.accepted),
            "metadata": dict(self.metadata),
        }

    def report(self) -> str:
        """A human-readable markdown scorecard for the run."""
        pct = self.improvement * 100
        lines = [
            f"# Skill optimization report — {self.optimizer}",
            "",
            f"- **Baseline (seed) score:** {self.baseline_score:.3f}",
            f"- **Best score:** {self.best_score:.3f}",
            f"- **Absolute improvement:** {pct:+.1f} points",
            f"- **Candidates evaluated:** {len(self.history)} "
            f"({sum(1 for c in self.history if c.accepted)} accepted)",
            "",
            "## Candidate history",
            "",
            "| iter | train | val | accepted | note |",
            "| ---: | ----: | --: | :------: | :--- |",
        ]
        for c in self.history:
            mark = "✓" if c.accepted else "·"
            note = (c.note or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {c.iteration} | {c.train_score:.3f} | {c.val_score:.3f} | {mark} | {note} |"
            )
        return "\n".join(lines) + "\n"
