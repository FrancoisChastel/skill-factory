"""Pluggable optimizer backends.

All optimizers implement the same :class:`Optimizer` protocol — take a seed
skill + train/val datasets + a harness + a metric, return an
:class:`~skill_factory.core.result.OptimizationResult`. Backends:

    * ``llm_loop``  — self-contained reflective loop (no heavy deps). Default.
    * ``dspy_gepa`` — DSPy GEPA reflective/evolutionary optimizer.
    * ``skillopt``  — Microsoft SkillOpt (trajectory-driven, validation-gated).
"""

from skill_factory.optimizers.base import (
    Optimizer,
    OptimizerConfig,
    get_optimizer,
    register_optimizer,
)
from skill_factory.optimizers.llm_loop import LLMLoopOptimizer

__all__ = [
    "Optimizer",
    "OptimizerConfig",
    "get_optimizer",
    "register_optimizer",
    "LLMLoopOptimizer",
]
