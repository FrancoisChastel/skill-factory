"""Metrics score a rollout and return a score plus textual feedback.

Three families, freely combined via :class:`CompositeMetric`:

    * :class:`GoldenMetric`       — compare output to a labeled expected answer.
    * :class:`ProgrammaticMetric` — deterministic checks (regex, JSON, code).
    * :class:`LLMJudgeMetric`     — rubric-based scoring by a judge model.
"""

from skill_factory.metrics.base import Metric
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.metrics.programmatic import ProgrammaticMetric, checks
from skill_factory.metrics.llm_judge import LLMJudgeMetric, RubricCriterion
from skill_factory.metrics.composite import CompositeMetric, WeightedMetric

__all__ = [
    "Metric",
    "GoldenMetric",
    "MatchMode",
    "ProgrammaticMetric",
    "checks",
    "LLMJudgeMetric",
    "RubricCriterion",
    "CompositeMetric",
    "WeightedMetric",
]
