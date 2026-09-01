"""Optimizer protocol, shared config, and a lazy registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from skill_factory.core.result import OptimizationResult
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset
from skill_factory.harness.base import Harness
from skill_factory.metrics.base import Metric


@dataclass(frozen=True)
class OptimizerConfig:
    """Common optimization hyperparameters (deep-learning analogues).

    Backend-specific knobs go in :attr:`params`.
    """

    rounds: int = 6  # epochs / GEPA iterations budget
    minibatch_size: int = 4  # train tasks sampled per reflective step
    max_edits_per_round: int = 1  # "learning rate": candidates proposed per round
    patience: int = 3  # early-stop after N non-improving rounds
    seed: int = 0
    max_workers: int = 1  # parallel rollouts during evaluation
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")


@runtime_checkable
class Optimizer(Protocol):
    """Mutates a seed skill to maximize a metric on held-out validation tasks."""

    name: str

    def optimize(
        self,
        seed: Skill,
        trainset: Dataset,
        valset: Dataset,
        harness: Harness,
        metric: Metric,
    ) -> OptimizationResult:
        ...


# --- Registry -------------------------------------------------------------
# Constructors are registered lazily so importing this module never pulls in
# dspy / skillopt. Each factory takes (config, **kwargs) and returns an Optimizer.

_REGISTRY: dict[str, Callable[..., Optimizer]] = {}


def register_optimizer(name: str, factory: Callable[..., Optimizer]) -> None:
    """Register an optimizer factory under ``name``."""
    _REGISTRY[name] = factory


def get_optimizer(name: str, config: OptimizerConfig | None = None, **kwargs: Any) -> Optimizer:
    """Instantiate a registered optimizer by name.

    Raises:
        ValueError: for an unknown optimizer name.
    """
    key = name.strip().lower()
    if key not in _REGISTRY:
        _ensure_defaults_registered()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key](config=config or OptimizerConfig(), **kwargs)


def available_optimizers() -> list[str]:
    _ensure_defaults_registered()
    return sorted(_REGISTRY)


def _ensure_defaults_registered() -> None:
    """Import built-in optimizer modules so they self-register (idempotent)."""
    # Local imports avoid a circular import at module load time.
    from skill_factory.optimizers import llm_loop  # noqa: F401

    try:  # optional heavy backends — register only if importable
        from skill_factory.optimizers import dspy_gepa  # noqa: F401
    except Exception:  # pragma: no cover - depends on optional extras
        pass
    try:
        from skill_factory.optimizers import skillopt as _skillopt  # noqa: F401
    except Exception:  # pragma: no cover - depends on optional extras
        pass
