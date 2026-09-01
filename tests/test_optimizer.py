"""End-to-end tests for the reflective loop optimizer (fully offline)."""

from __future__ import annotations

from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.harness.callable_harness import CallableHarness
from skill_factory.llm.client import FakeLLMClient
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.optimizers.base import OptimizerConfig
from skill_factory.optimizers.llm_loop import LLMLoopOptimizer

# A harness whose output is correct only when the skill body contains the marker.
# This lets an offline "optimizer LLM" demonstrably improve the score by adding it.
_MARKER = "STRUCTURED-OUTPUT"


def _harness() -> CallableHarness:
    def run(skill: Skill, task: Task) -> str:
        return task.expected if _MARKER in skill.body else "wrong"

    return CallableHarness(run)


def _dataset() -> Dataset:
    return Dataset([Task(id=f"t{i}", input=f"in{i}", expected=f"ans{i}") for i in range(8)])


def _seed() -> Skill:
    return Skill(name="s", description="d", body="Answer the question.")


def _improving_client() -> FakeLLMClient:
    body = f"Answer the question.\n\nAlways produce {_MARKER}."
    response = f"<RATIONALE>add marker</RATIONALE>\n<NEW_SKILL_BODY>\n{body}\n</NEW_SKILL_BODY>"
    return FakeLLMClient(responder=lambda prompt, system: response)


def test_loop_improves_and_gates_on_validation():
    ds = _dataset()
    train, val = ds.split(0.5, seed=0)
    opt = LLMLoopOptimizer(
        _improving_client(),
        config=OptimizerConfig(rounds=3, minibatch_size=2, patience=3),
    )
    result = opt.optimize(_seed(), train, val, _harness(), GoldenMetric(MatchMode.EXACT))

    assert result.baseline_score == 0.0
    assert result.best_score == 1.0
    assert result.improvement == 1.0
    assert _MARKER in result.best_skill.body
    assert any(c.accepted and c.iteration > 0 for c in result.history)


def test_loop_rejects_non_improving_edit():
    ds = _dataset()
    train, val = ds.split(0.5, seed=0)
    # Optimizer proposes a body WITHOUT the marker -> harness stays wrong -> gate rejects.
    useless = "<NEW_SKILL_BODY>\nAnswer carefully and precisely.\n</NEW_SKILL_BODY>"
    client = FakeLLMClient(responder=lambda p, s: useless)
    opt = LLMLoopOptimizer(client, config=OptimizerConfig(rounds=2, minibatch_size=2, patience=5))
    result = opt.optimize(_seed(), train, val, _harness(), GoldenMetric(MatchMode.EXACT))

    assert result.best_score == 0.0
    assert result.best_skill.body == _seed().body  # unchanged
    assert all(not c.accepted for c in result.history if c.iteration > 0)


def test_loop_handles_unparseable_proposal():
    ds = _dataset()
    train, val = ds.split(0.5, seed=0)
    client = FakeLLMClient(responder=lambda p, s: "no markers here")
    opt = LLMLoopOptimizer(client, config=OptimizerConfig(rounds=2, minibatch_size=2, patience=1))
    result = opt.optimize(_seed(), train, val, _harness(), GoldenMetric(MatchMode.EXACT))
    assert result.best_score == 0.0
    assert "no valid proposal" in result.history[-1].note


def test_report_renders_markdown():
    ds = _dataset()
    train, val = ds.split(0.5, seed=0)
    opt = LLMLoopOptimizer(_improving_client(), config=OptimizerConfig(rounds=1, minibatch_size=2))
    result = opt.optimize(_seed(), train, val, _harness(), GoldenMetric(MatchMode.EXACT))
    report = result.report()
    assert "# Skill optimization report" in report
    assert "Baseline" in report
