"""Policies: how answers become a verdict, replayed offline on cached answers.

The default policy flags an example when its score reaches the threshold. A
real pipeline usually does more: skill-scanner's judge doubts a static finding
when a capability question answers low, and adds its own finding when the
threat score is high. Write that as a function and every variant can be scored
on every split from the cache, without a call:

    # policy.py
    LEVELS = ("pass", "warn", "block")

    def judge(example, answers, score, threshold):
        findings = example.metadata.get("findings", [])
        ...
        return "warn"

    skill-factory lab simulate -c config.yaml --policy policy.py:judge --policy policy.py:judge_with_confirm

That is how confirming was dropped from jev's judge: replayed on training and
validation, it added a false block and caught nothing the other rules missed.

A policy file is Python and runs with your permissions, like the config that names it.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.questions import sha24

PolicyFn = Callable[[Example, Mapping[str, float], "float | None", "float | None"], str]
DEFAULT_LEVELS = ("pass", "flag")


@dataclass(frozen=True)
class Policy:
    """A named verdict function and its ordered levels (the first means "not detected")."""

    name: str
    fn: PolicyFn
    levels: tuple[str, ...] = DEFAULT_LEVELS

    def __post_init__(self) -> None:
        if len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
            raise ValueError("a policy needs at least two distinct levels")

    def verdict(
        self, example: Example, answers: Mapping[str, float], score: float | None, threshold: float | None
    ) -> str:
        v = self.fn(example, answers, score, threshold)
        if v not in self.levels:
            raise ValueError(f"policy {self.name!r} returned {v!r}, not one of {self.levels}")
        return v

    def at_least(self, verdict: str, level: str) -> bool:
        return self.levels.index(verdict) >= self.levels.index(level)

    @property
    def detect_levels(self) -> tuple[str, ...]:
        return self.levels[1:]


def _flag_at_threshold(
    example: Example, answers: Mapping[str, float], score: float | None, threshold: float | None
) -> str:
    return "flag" if score is not None and threshold is not None and score >= threshold else "pass"


THRESHOLD_POLICY = Policy("threshold", _flag_at_threshold)


def load_policy(ref: str, base_dir: str | Path = ".") -> Policy:
    """Load ``path/to/file.py:function`` (relative to ``base_dir``) as a Policy.

    The levels come from the function's ``levels`` attribute, else the module's
    ``LEVELS``, else ``("pass", "flag")``. The object may also be a Policy.
    """
    if ":" not in ref:
        raise ValueError(f"policy must be 'file.py:function', got {ref!r}")
    file_part, attr = ref.rsplit(":", 1)
    path = Path(file_part)
    if not path.is_absolute():
        path = Path(base_dir) / path
    if not path.exists():
        raise FileNotFoundError(f"policy file not found: {path}")
    module_name = f"_skill_factory_policy_{sha24(str(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load policy file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    obj: Any = getattr(module, attr, None)
    if obj is None:
        raise ValueError(f"{path} has no {attr!r}")
    if isinstance(obj, Policy):
        return obj
    if not callable(obj):
        raise ValueError(f"{path}:{attr} is not callable")
    levels = getattr(obj, "levels", None) or getattr(module, "LEVELS", None) or DEFAULT_LEVELS
    return Policy(name=attr, fn=cast(PolicyFn, obj), levels=tuple(levels))
