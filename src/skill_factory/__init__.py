"""Skill Factory — scientifically optimize portable agent skills (SKILL.md).

The factory is built around five primitives that every optimizer shares:

    Skill      the trainable text (a SKILL.md document)
    Dataset    the tasks the skill must handle
    Harness    runs a skill on a task and returns a Rollout
    Metric     scores a rollout and returns *textual feedback*
    Optimizer  mutates the skill against the metric to maximize the score

Optimizer backends (custom LLM loop, DSPy/GEPA, Microsoft SkillOpt) are just
adapters over these primitives.
"""

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

__version__ = "0.1.0"

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
    "__version__",
]
