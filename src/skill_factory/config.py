"""Declarative run configuration (YAML) and validation.

A config fully describes an optimization run: which skill and dataset, which
providers (target / optimizer / judge), which metric mixture, and which
optimizer + hyperparameters. The :mod:`skill_factory.builder` turns this into
live objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProviderConfig:
    """An LLM provider + model (see skill_factory.llm.factory.SUPPORTED_PROVIDERS)."""

    name: str = "anthropic"
    model: str = "claude-sonnet-5"
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, default_model: str) -> "ProviderConfig":
        data = data or {}
        return cls(
            name=str(data.get("name", "anthropic")),
            model=str(data.get("model", default_model)),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
        )


@dataclass(frozen=True)
class RunConfig:
    """The full, validated configuration for one optimization run."""

    skill_path: Path
    dataset_path: Path
    val_fraction: float
    seed: int
    target: ProviderConfig
    optimizer_provider: ProviderConfig
    judge_provider: ProviderConfig
    optimizer: dict[str, Any]
    metric: dict[str, Any]
    harness: dict[str, Any]
    output: dict[str, Any]
    base_dir: Path = field(default=Path("."))

    def resolve(self, path: str | Path) -> Path:
        """Resolve a config-relative path against the config file's directory."""
        p = Path(path)
        return p if p.is_absolute() else (self.base_dir / p)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from YAML.

    Raises:
        FileNotFoundError: if the config file is missing.
        ValueError: if required fields are absent or malformed.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a YAML mapping")

    _require(raw, "skill")
    _require(raw, "dataset")

    base_dir = cfg_path.parent
    target = ProviderConfig.from_dict(
        raw.get("provider") or raw.get("target"),
        default_model=os.getenv("SKILL_FACTORY_TARGET_MODEL", "claude-sonnet-5"),
    )
    optimizer_provider = ProviderConfig.from_dict(
        raw.get("optimizer_provider"),
        default_model=os.getenv("SKILL_FACTORY_OPTIMIZER_MODEL", "claude-opus-4-8"),
    )
    judge_provider = ProviderConfig.from_dict(
        raw.get("judge_provider"),
        default_model=os.getenv("SKILL_FACTORY_JUDGE_MODEL", target.model),
    )

    val_fraction = float(raw.get("val_fraction", 0.3))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    return RunConfig(
        skill_path=base_dir / str(raw["skill"]),
        dataset_path=base_dir / str(raw["dataset"]),
        val_fraction=val_fraction,
        seed=int(raw.get("seed", 0)),
        target=target,
        optimizer_provider=optimizer_provider,
        judge_provider=judge_provider,
        optimizer=dict(raw.get("optimizer", {"name": "llm_loop"})),
        metric=dict(raw.get("metric", {})),
        harness=dict(raw.get("harness", {})),
        output=dict(raw.get("output", {})),
        base_dir=base_dir,
    )


def _require(data: dict[str, Any], key: str) -> None:
    if key not in data:
        raise ValueError(f"Config is missing required field: {key!r}")
