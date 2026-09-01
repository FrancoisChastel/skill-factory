"""Evaluate, export, provider factory, and config/builder integration tests."""

from __future__ import annotations

import pytest

from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.evaluate import evaluate_skill
from skill_factory.export import export_skill, slugify
from skill_factory.harness.callable_harness import CallableHarness
from skill_factory.llm.client import FakeLLMClient
from skill_factory.llm.factory import SUPPORTED_PROVIDERS, make_client
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.optimizers.base import available_optimizers


# --- evaluate -------------------------------------------------------------

def test_evaluate_orders_results_and_scores():
    ds = Dataset([Task(id=f"t{i}", input=str(i), expected=str(i)) for i in range(5)])
    harness = CallableHarness(lambda skill, task: task.input)  # echo -> exact match
    ev = evaluate_skill(Skill(name="s", description="d", body="b"), ds, harness,
                        GoldenMetric(MatchMode.EXACT))
    assert ev.score == 1.0
    assert [i.task.id for i in ev.items] == [f"t{i}" for i in range(5)]


def test_evaluate_parallel_preserves_order():
    ds = Dataset([Task(id=f"t{i}", input=str(i), expected=str(i)) for i in range(12)])
    harness = CallableHarness(lambda skill, task: task.input)
    ev = evaluate_skill(Skill(name="s", description="d", body="b"), ds, harness,
                        GoldenMetric(MatchMode.EXACT), max_workers=4)
    assert [i.task.id for i in ev.items] == [f"t{i}" for i in range(12)]


# --- export (npx skills) --------------------------------------------------

def test_export_creates_skill_dir(tmp_path):
    skill = Skill(name="My Skill!", description="does things", body="do it")
    result = export_skill(skill, tmp_path)
    assert result.skill_dir == tmp_path / "my-skill"
    assert result.skill_md.exists()
    assert "name: My Skill!" in result.skill_md.read_text()


def test_export_collection_layout(tmp_path):
    skill = Skill(name="alpha", description="d", body="b")
    result = export_skill(skill, tmp_path, as_collection=True)
    assert result.skill_md == tmp_path / "skills" / "alpha" / "SKILL.md"


def test_export_requires_description(tmp_path):
    skill = Skill(name="x", description="   ", body="b")
    with pytest.raises(ValueError, match="description"):
        export_skill(skill, tmp_path)


def test_export_bundles_extra_files(tmp_path):
    skill = Skill(name="x", description="d", body="b")
    result = export_skill(skill, tmp_path, extra_files={"reference.md": "ref"})
    assert (result.skill_dir / "reference.md").read_text() == "ref"


def test_slugify():
    assert slugify("Hello World 2!") == "hello-world-2"
    with pytest.raises(ValueError):
        slugify("!!!")


# --- provider factory -----------------------------------------------------

def test_make_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        make_client("nonsense", "model-x")


def test_supported_providers_includes_common():
    for p in ("anthropic", "openai", "ollama", "openrouter"):
        assert p in SUPPORTED_PROVIDERS


# --- registry -------------------------------------------------------------

def test_registry_lists_backends():
    names = available_optimizers()
    assert "llm_loop" in names
    assert "skillopt" in names  # subprocess bridge, always registerable


# --- builder --------------------------------------------------------------

def test_build_metric_from_config_with_fake_judge():
    from skill_factory.builder import build_metric
    from skill_factory.core.rollout import Rollout

    judge = FakeLLMClient(responses=['{"scores": {"correctness": 10}, "rationale": "ok"}'])
    metric = build_metric(
        {
            "golden": {"mode": "json_equal", "weight": 2.0},
            "programmatic": {"checks": ["is_valid_json", {"json_has_keys": ["a"]}]},
            "judge": {"weight": 1.0, "criteria": [{"name": "correctness", "description": "x"}]},
        },
        judge,
    )
    task = Task(id="t", input="in", expected='{"a": 1}')
    rollout = Rollout(task=task, output='{"a": 1}')
    res = metric.evaluate(task, rollout)
    assert res.score > 0.9  # all three components near-perfect


def test_build_metric_defaults_to_golden_when_empty():
    from skill_factory.builder import build_metric
    from skill_factory.core.rollout import Rollout

    metric = build_metric({}, None)
    task = Task(id="t", input="in", expected="hello")
    assert metric.evaluate(task, Rollout(task=task, output="hello")).score == 1.0


def test_load_config_and_run_offline(tmp_path, monkeypatch):
    """Load the real example config shape and run a full optimization offline."""
    from skill_factory import builder
    from skill_factory.config import load_config

    # Write a minimal config + data.
    (tmp_path / "seed_skill.md").write_text(
        "---\nname: s\ndescription: d\n---\nAnswer.\n"
    )
    (tmp_path / "data.jsonl").write_text(
        "\n".join(
            f'{{"id":"t{i}","input":"{i}","expected":"{i}"}}' for i in range(6)
        )
    )
    (tmp_path / "config.yaml").write_text(
        "skill: seed_skill.md\n"
        "dataset: data.jsonl\n"
        "val_fraction: 0.5\n"
        "optimizer:\n  name: llm_loop\n  rounds: 1\n  minibatch_size: 2\n"
        "metric:\n  golden:\n    mode: exact\n"
    )

    # Patch client construction so no network/keys are needed.
    from skill_factory.harness.callable_harness import CallableHarness

    monkeypatch.setattr(builder, "build_client", lambda provider: FakeLLMClient(responses=[""]))
    monkeypatch.setattr(
        builder, "build_harness",
        lambda client, cfg: CallableHarness(lambda skill, task: task.input),
    )

    config = load_config(tmp_path / "config.yaml")
    result = builder.run_optimization(config)
    assert result.best_score == 1.0  # echo harness matches expected exactly
