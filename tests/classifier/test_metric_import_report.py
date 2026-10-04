"""jev as a skill-mode metric, the skill-scanner lab importer, and report rendering."""

from __future__ import annotations

import json

import pytest

from skill_factory.builder import build_metric
from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.dataset import load_examples
from skill_factory.classifier.lab import Lab
from skill_factory.classifier.objective import Objective
from skill_factory.classifier.optimizer import ClassifierOptimizer
from skill_factory.classifier.questions import Question
from skill_factory.classifier.report import before_after_table, classifier_report, coverage_table, splits_svg
from skill_factory.classifier.scanner_import import import_scanner_lab
from skill_factory.classifier.splits import SplitPlan
from skill_factory.classifier.systemone import FakeSystemOne
from skill_factory.core.rollout import Rollout
from skill_factory.core.task import Task
from skill_factory.metrics.systemone import Check, SystemOneMetric

from .conftest import bank, responder

STRICT = Check("strict_json", "Is the OUTPUT strict JSON?", "Yes: strict JSON.", "No: prose or not JSON.")
GROUNDED = Check("grounded", "Is every value grounded in the INPUT?", "Yes: grounded.", "No: invented.", weight=3)


def _json_judge(state: str, qid: str, q: Question) -> float:
    output = state.split("=== OUTPUT ===\n", 1)[1]
    if qid == "strict_json":
        return 0.97 if output.lstrip().startswith("{") else 0.03
    return 0.9


def test_systemone_metric_scores_weighted_probabilities_and_explains_failures():
    client = FakeSystemOne(_json_judge)
    metric = SystemOneMetric(client, [STRICT, GROUNDED])
    task = Task(id="t", input="invoice text")
    good = metric.evaluate(task, Rollout(task, '{"total": 3}'))
    bad = metric.evaluate(task, Rollout(task, "Here is the JSON: {...}"))
    assert good.score == pytest.approx((0.97 + 3 * 0.9) / 4)
    assert bad.score < good.score and "strict_json" in bad.feedback and "prose" in bad.feedback
    assert good.feedback == "all checks pass"
    assert len(client.requests) == 2 and set(client.requests[0][1]) == {"strict_json", "grounded"}


def test_systemone_metric_caches_threshold_mode_and_failures():
    client = FakeSystemOne(_json_judge)
    metric = SystemOneMetric(client, [STRICT], threshold=0.5)
    task = Task(id="t", input="x", expected="ref")
    assert metric.evaluate(task, Rollout(task, "{}")).score == 1.0
    assert metric.evaluate(task, Rollout(task, "{}")).score == 1.0
    assert len(client.requests) == 1  # second call came from the cache
    assert metric.evaluate(task, Rollout.failed(task, "boom")).score == 0.0
    broken = SystemOneMetric(FakeSystemOne(_json_judge, fail_when=lambda s: True), [STRICT])
    assert "failed" in broken.evaluate(task, Rollout(task, "{}")).feedback
    with_ref = SystemOneMetric(FakeSystemOne(_json_judge), [STRICT], include_expected=True)
    assert "=== REFERENCE ===" in with_ref._state(task, Rollout(task, "{}"))
    with pytest.raises(ValueError):
        SystemOneMetric(client, [])
    with pytest.raises(ValueError):
        SystemOneMetric(client, [STRICT, STRICT])
    with pytest.raises(ValueError):
        Check("x", "q", "t", "f", weight=0)


def test_builder_wires_the_systemone_metric(stub_server, tmp_path):
    url, _ = stub_server
    metric = build_metric(
        {"systemone": {"provider": "custom", "base_url": url, "cache": str(tmp_path / "c.jsonl"),
                       "checks": [{"name": "intent", "question": "Does it show bad intent?"}]}},
        None,
    )
    task = Task(id="t", input="item 3 INTENT:bad")
    assert metric.evaluate(task, Rollout(task, "ok")).score > 0.5
    assert (tmp_path / "c.jsonl").exists()
    with pytest.raises(ValueError, match="checks"):
        build_metric({"systemone": {"provider": "custom", "base_url": url}}, None)


# --- skill-scanner import -----------------------------------------------------------

def _scanner_lab(tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    q = Question("Is it malicious?", "Yes: malicious.", "No: benign.")
    rows = [
        {"dir": "/corpora/m/a", "corpus": "malicious/x", "label": "malicious", "split": "train", "kind": "skill",
         "chars": 100, "stateHash": "s1", "findings": [{"ruleId": "r", "category": "network", "severity": "high",
                                                         "confidence": "high", "hard": False}]},
        {"dir": "/corpora/b/c", "corpus": "benign/y", "label": "benign", "split": "val", "kind": "skill",
         "chars": 90, "stateHash": "s2", "findings": []},
        {"dir": "/corpora/b/d", "corpus": "benign/y", "label": "benign", "split": "train", "kind": "skill",
         "skipped": "src/x.py was truncated while collecting; not sent to the judge", "findings": []},
        {"dir": "/corpora/m/e", "corpus": "malicious/x", "label": "malicious", "split": "train", "kind": "skill",
         "skipped": "skill text is 120000 characters, over the judge's 96000 budget; skipped rather than truncated",
         "findings": []},
    ]
    (lab / "bundles-96000.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    answers = [{"s": "s1", "q": q.hash, "m": "jev-latest", "p": 0.91}, {"s": "s2", "q": q.hash, "m": "jev-latest", "p": 0.02}]
    (lab / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in answers) + "\n")
    (tmp_path / "q.json").write_text(json.dumps({"malicious": {"instructions": q.instructions, "true": q.true, "false": q.false}}))
    return lab, q


def test_import_scanner_lab_keeps_splits_findings_and_cached_answers(tmp_path):
    lab_dir, q = _scanner_lab(tmp_path)
    res = import_scanner_lab(lab_dir, tmp_path / "ws", questions=tmp_path / "q.json")
    assert res.examples == 4 and res.pool is not None
    exs = {e.id: e for e in load_examples(res.dataset, positive="malicious", negative="benign")}
    assert set(exs) == {"m/a", "b/c", "b/d", "m/e"}
    # The size is kept so the workspace's own budget decides; other reasons stay skips.
    assert exs["m/e"].chars == 120_000 and exs["m/e"].status(96_000) == "over budget"
    assert exs["m/e"].status(200_000) == "no state"
    assert exs["m/a"].label and exs["m/a"].split == "train" and exs["m/a"].metadata["findings"][0]["category"] == "network"
    assert exs["b/d"].status(96_000).startswith("skipped")
    lab = Lab(None, AnswerCache(res.answers), model="jev-latest")
    table = lab.table(list(exs.values()), {"malicious": q}, budget=96_000)
    assert table.p("m/a", "malicious") == 0.91 and table.p("b/c", "malicious") == 0.02
    with pytest.raises(FileNotFoundError, match="not a skill-scanner lab"):
        import_scanner_lab(tmp_path, tmp_path / "x")


# --- reports ----------------------------------------------------------------------------

def test_report_tables_and_figure(examples, seed_probeset):
    by = SplitPlan().assign(examples)
    lab = Lab(FakeSystemOne(responder), AnswerCache(None))
    result = ClassifierOptimizer(lab, Objective(fpr_budget=0.05)).optimize(
        seed_probeset, {"train": by["train"], "val": by["val"]}, bank())
    md = classifier_report(result, seal_summary="not frozen", cost_per_pass=0.01,
                           statuses={"train": {"judged": 10, "over budget": 2}}, sealed_splits=["test"])
    assert "| test | sealed" in md and "held-out | sealed" not in md
    assert "## Coverage" in md and "| training | 10 | 2 |" in md
    assert "## Pareto front" in md and "0.0100" in md
    svg = splits_svg(result.before, result.after)
    assert svg.startswith("<svg") and "training" in svg and "validation" in svg
    assert splits_svg({}, {}).startswith("<svg")
    table = before_after_table({}, result.after)
    assert "-> **" not in table and "**" in table
    assert coverage_table({}) .startswith("| Split")
