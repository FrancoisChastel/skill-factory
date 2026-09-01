"""Tests for golden / programmatic / judge / composite metrics."""

from __future__ import annotations

import pytest

from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.llm.client import FakeLLMClient
from skill_factory.metrics.base import MetricNotApplicable
from skill_factory.metrics.composite import CompositeMetric, WeightedMetric
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.metrics.llm_judge import LLMJudgeMetric, RubricCriterion
from skill_factory.metrics.programmatic import ProgrammaticMetric, checks


def _task(expected=None):
    return Task(id="t", input="in", expected=expected)


def _rollout(output, task=None):
    return Rollout(task=task or _task(), output=output)


# --- golden ---------------------------------------------------------------

def test_golden_exact():
    m = GoldenMetric(MatchMode.EXACT)
    assert m.evaluate(_task("hello"), _rollout("hello")).score == 1.0
    assert m.evaluate(_task("hello"), _rollout("Hello")).score == 0.0


def test_golden_normalized_ignores_case_and_space():
    m = GoldenMetric(MatchMode.NORMALIZED)
    assert m.evaluate(_task("Hello World"), _rollout("hello   world")).score == 1.0


def test_golden_token_f1_partial():
    m = GoldenMetric(MatchMode.TOKEN_F1)
    score = m.evaluate(_task("the quick brown fox"), _rollout("the quick red fox")).score
    assert 0.0 < score < 1.0


def test_golden_json_equal_partial_credit():
    m = GoldenMetric(MatchMode.JSON_EQUAL)
    task = _task('{"a": 1, "b": 2}')
    res = m.evaluate(task, _rollout('{"a": 1, "b": 99}'))
    assert 0.0 < res.score < 1.0  # one of two keys correct


def test_golden_json_tolerates_code_fence():
    m = GoldenMetric(MatchMode.JSON_EQUAL)
    task = _task('{"a": 1}')
    res = m.evaluate(task, _rollout("```json\n{\"a\": 1}\n```"))
    assert res.score == 1.0


def test_golden_numeric_tolerance():
    m = GoldenMetric(MatchMode.NUMERIC)
    assert m.evaluate(_task("Total: 10.00"), _rollout("it is 10")).score == 1.0


def test_golden_not_applicable_without_expected():
    m = GoldenMetric()
    with pytest.raises(MetricNotApplicable):
        m.evaluate(_task(None), _rollout("anything"))


# --- programmatic ---------------------------------------------------------

def test_programmatic_valid_json_and_keys():
    m = ProgrammaticMetric([checks.is_valid_json(), checks.json_has_keys("a", "b")])
    res = m.evaluate(_task(), _rollout('{"a": 1, "b": 2}'))
    assert res.score == 1.0


def test_programmatic_partial_keys():
    m = ProgrammaticMetric([checks.json_has_keys("a", "b")])
    res = m.evaluate(_task(), _rollout('{"a": 1}'))
    assert res.score == 0.5


def test_programmatic_regex_and_contains():
    m = ProgrammaticMetric([checks.matches_regex(r"\d+"), checks.contains("cat")])
    assert m.evaluate(_task(), _rollout("a cat and 3 dogs")).score == 1.0
    assert m.evaluate(_task(), _rollout("no animals")).score == 0.0


def test_programmatic_not_contains():
    m = ProgrammaticMetric([checks.not_contains("error")])
    assert m.evaluate(_task(), _rollout("all good")).score == 1.0
    assert m.evaluate(_task(), _rollout("an error occurred")).score == 0.0


def test_failed_rollout_scores_zero():
    m = ProgrammaticMetric([checks.is_valid_json()])
    res = m.evaluate(_task(), Rollout.failed(_task(), "boom"))
    assert res.score == 0.0


# --- judge ----------------------------------------------------------------

def test_llm_judge_parses_scores():
    client = FakeLLMClient(
        responses=['{"scores": {"correctness": 8, "format": 10}, "rationale": "good"}']
    )
    m = LLMJudgeMetric(
        client,
        [RubricCriterion("correctness", "is it right", 3),
         RubricCriterion("format", "is it json", 1)],
    )
    res = m.evaluate(_task(), _rollout("output"))
    # (0.8*3 + 1.0*1) / 4 = 0.85
    assert res.score == pytest.approx(0.85, abs=1e-6)
    assert "good" in res.feedback


def test_llm_judge_handles_unparseable():
    client = FakeLLMClient(responses=["not json at all"])
    m = LLMJudgeMetric(client, [RubricCriterion("x", "y")])
    assert m.evaluate(_task(), _rollout("o")).score == 0.0


# --- composite ------------------------------------------------------------

def test_composite_weighted_average():
    golden = GoldenMetric(MatchMode.EXACT)  # will score 1.0
    prog = ProgrammaticMetric([checks.contains("z")])  # will score 0.0
    comp = CompositeMetric([WeightedMetric(golden, 3), WeightedMetric(prog, 1)])
    res = comp.evaluate(_task("hi"), _rollout("hi"))
    # (1.0*3 + 0.0*1) / 4 = 0.75
    assert res.score == pytest.approx(0.75)


def test_composite_skips_not_applicable():
    golden = GoldenMetric(MatchMode.EXACT)  # not applicable (no expected)
    prog = ProgrammaticMetric([checks.contains("hi")])  # scores 1.0
    comp = CompositeMetric([WeightedMetric(golden), WeightedMetric(prog)])
    res = comp.evaluate(_task(None), _rollout("hi there"))
    assert res.score == 1.0  # golden dropped, only programmatic counts
