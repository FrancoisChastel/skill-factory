"""Guard the shipped example configs: they must load, build, and score sanely."""

from __future__ import annotations

from pathlib import Path

import pytest

from skill_factory.builder import build_metric
from skill_factory.config import load_config
from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.llm.client import FakeLLMClient

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_CONFIGS = [
    _EXAMPLES / "invoice-extractor" / "config.yaml",
    _EXAMPLES / "invoice-extractor" / "config.fast.yaml",
    _EXAMPLES / "ticket-classifier" / "config.yaml",
]


@pytest.mark.parametrize("config_path", _CONFIGS, ids=lambda p: p.parent.name + "/" + p.name)
def test_example_config_loads_and_builds(config_path):
    config = load_config(config_path)
    # Seed skill and dataset referenced by the config must exist and parse.
    seed = Skill.load(config.skill_path)
    dataset = Dataset.from_jsonl(config.dataset_path)
    assert seed.name and len(dataset) >= 4

    judge = FakeLLMClient(responses=['{"scores": {"correctness": 9}, "rationale": "ok"}'])
    metric = build_metric(config.metric, judge)
    # Smoke: metric can score a rollout for the first task without raising.
    task = dataset[0]
    metric.evaluate(task, Rollout(task=task, output=task.expected or "x"))


def test_ticket_classifier_regex_rewards_bare_label():
    config = load_config(_EXAMPLES / "ticket-classifier" / "config.yaml")
    judge = FakeLLMClient(responses=['{"scores": {"correctness": 10, "format": 10}, "rationale": "ok"}'])
    metric = build_metric(config.metric, judge)

    task = Task(id="t", input="charged twice", expected="billing")
    clean = metric.evaluate(task, Rollout(task=task, output="billing"))
    noisy = metric.evaluate(task, Rollout(task=task, output="This ticket is about billing."))
    # A bare, correct label must outscore a correct-but-verbose answer.
    assert clean.score > noisy.score
    assert clean.score == pytest.approx(1.0)
