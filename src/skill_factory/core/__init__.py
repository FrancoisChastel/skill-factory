"""Core, dependency-light primitives shared by every optimizer backend."""

from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.core.rollout import Rollout
from skill_factory.core.result import (
    CandidateRecord,
    Evaluation,
    MetricResult,
    OptimizationResult,
    TaskEvaluation,
)

__all__ = [
    "Skill",
    "Task",
    "Dataset",
    "Rollout",
    "MetricResult",
    "TaskEvaluation",
    "Evaluation",
    "CandidateRecord",
    "OptimizationResult",
]
