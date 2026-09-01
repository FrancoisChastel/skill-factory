"""Turn a validated RunConfig into live objects and execute a run."""

from __future__ import annotations

from typing import Any

from skill_factory.config import ProviderConfig, RunConfig
from skill_factory.core.result import OptimizationResult
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset
from skill_factory.harness.anthropic_api import AnthropicHarness
from skill_factory.harness.base import Harness
from skill_factory.llm.client import LLMClient
from skill_factory.llm.factory import make_client
from skill_factory.metrics.base import Metric
from skill_factory.metrics.composite import CompositeMetric, WeightedMetric
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.metrics.llm_judge import LLMJudgeMetric, RubricCriterion
from skill_factory.metrics.programmatic import ProgrammaticMetric, checks
from skill_factory.optimizers.base import OptimizerConfig, Optimizer, get_optimizer


def build_client(provider: ProviderConfig) -> LLMClient:
    return make_client(
        provider.name, provider.model, api_key=provider.api_key, base_url=provider.base_url
    )


def build_harness(target_client: LLMClient, harness_cfg: dict[str, Any]) -> Harness:
    return AnthropicHarness(
        target_client,
        temperature=float(harness_cfg.get("temperature", 0.0)),
        max_tokens=int(harness_cfg.get("max_tokens", 2048)),
    )


def build_metric(metric_cfg: dict[str, Any], judge_client: LLMClient | None) -> Metric:
    """Assemble a CompositeMetric from golden / programmatic / judge sub-configs."""
    components: list[WeightedMetric] = []

    if "golden" in metric_cfg:
        components.append(_build_golden(metric_cfg["golden"]))
    if "programmatic" in metric_cfg:
        components.append(_build_programmatic(metric_cfg["programmatic"]))
    if "judge" in metric_cfg:
        if judge_client is None:
            raise ValueError("metric.judge is configured but no judge client was built")
        components.append(_build_judge(metric_cfg["judge"], judge_client))

    if not components:
        # Sensible default: normalized golden match.
        components.append(WeightedMetric(GoldenMetric(MatchMode.NORMALIZED)))
    return CompositeMetric(components)


def build_optimizer(
    optimizer_cfg: dict[str, Any],
    *,
    optimizer_client: LLMClient | None,
    optimizer_provider: ProviderConfig,
) -> Optimizer:
    name = str(optimizer_cfg.get("name", "llm_loop"))
    config = OptimizerConfig(
        rounds=int(optimizer_cfg.get("rounds", 6)),
        minibatch_size=int(optimizer_cfg.get("minibatch_size", 4)),
        max_edits_per_round=int(optimizer_cfg.get("max_edits_per_round", 1)),
        patience=int(optimizer_cfg.get("patience", 3)),
        seed=int(optimizer_cfg.get("seed", 0)),
        max_workers=int(optimizer_cfg.get("max_workers", 1)),
        params=dict(optimizer_cfg.get("params", {})),
    )

    if name == "llm_loop":
        if optimizer_client is None:
            raise ValueError("llm_loop optimizer needs an optimizer_client")
        return get_optimizer(
            name,
            config=config,
            optimizer_client=optimizer_client,
            goal=str(optimizer_cfg.get("goal", "")),
            min_delta=float(optimizer_cfg.get("min_delta", 0.0)),
        )
    if name == "dspy_gepa":
        lm = optimizer_cfg.get("lm") or f"{optimizer_provider.name}/{optimizer_provider.model}"
        return get_optimizer(
            name,
            config=config,
            lm=lm,
            reflection_lm=optimizer_cfg.get("reflection_lm", lm),
            auto=str(optimizer_cfg.get("auto", "light")),
            goal=str(optimizer_cfg.get("goal", "")),
        )
    # skillopt and any other registered backend take only config.
    return get_optimizer(name, config=config)


def run_optimization(config: RunConfig) -> OptimizationResult:
    """Execute a full run from a RunConfig and return its result."""
    seed = Skill.load(config.skill_path)
    dataset = Dataset.from_jsonl(config.dataset_path)
    trainset, valset = dataset.split(config.val_fraction, seed=config.seed)

    target_client = build_client(config.target)
    harness = build_harness(target_client, config.harness)

    judge_client = build_client(config.judge_provider) if "judge" in config.metric else None
    metric = build_metric(config.metric, judge_client)

    optimizer_name = str(config.optimizer.get("name", "llm_loop"))
    optimizer_client = (
        build_client(config.optimizer_provider) if optimizer_name == "llm_loop" else None
    )
    optimizer = build_optimizer(
        config.optimizer,
        optimizer_client=optimizer_client,
        optimizer_provider=config.optimizer_provider,
    )

    return optimizer.optimize(seed, trainset, valset, harness, metric)


# --- sub-builders ---------------------------------------------------------

def _weight(cfg: dict[str, Any]) -> float:
    return float(cfg.get("weight", 1.0))


def _build_golden(cfg: dict[str, Any]) -> WeightedMetric:
    mode = cfg.get("mode", "normalized")
    return WeightedMetric(GoldenMetric(MatchMode(mode)), weight=_weight(cfg))


def _build_programmatic(cfg: dict[str, Any]) -> WeightedMetric:
    check_fns = [_build_check(entry) for entry in cfg.get("checks", [])]
    if not check_fns:
        raise ValueError("metric.programmatic requires a non-empty 'checks' list")
    return WeightedMetric(ProgrammaticMetric(check_fns), weight=_weight(cfg))


def _build_check(entry: Any):
    """Build a Check from 'name' or {'name': args}."""
    if isinstance(entry, str):
        return getattr(checks, entry)()
    if isinstance(entry, dict) and len(entry) == 1:
        (method, args), = entry.items()
        factory = getattr(checks, method, None)
        if factory is None:
            raise ValueError(f"unknown programmatic check: {method!r}")
        if isinstance(args, list):
            return factory(*args)
        if isinstance(args, dict):
            return factory(**args)
        return factory(args)
    raise ValueError(f"invalid check spec: {entry!r}")


def _build_judge(cfg: dict[str, Any], client: LLMClient) -> WeightedMetric:
    criteria = [
        RubricCriterion(
            name=str(c["name"]),
            description=str(c.get("description", "")),
            weight=float(c.get("weight", 1.0)),
        )
        for c in cfg.get("criteria", [])
    ]
    if not criteria:
        raise ValueError("metric.judge requires a non-empty 'criteria' list")
    return WeightedMetric(LLMJudgeMetric(client, criteria), weight=_weight(cfg))
