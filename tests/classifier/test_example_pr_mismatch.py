"""Guard the shipped classifier example: it loads, splits honestly, and runs end to end."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from skill_factory.classifier.config import load_classifier_config
from skill_factory.classifier.questions import Question
from skill_factory.classifier.systemone import FakeSystemOne
from skill_factory.classifier.workspace import Workspace, run_classifier

from .conftest import jitter

_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "pr-mismatch" / "config.yaml"


def test_example_splits_put_both_classes_in_every_split():
    ws = Workspace(load_classifier_config(_CONFIG))
    for split in ("train", "val", "test"):
        labels = [ex.label for ex in ws.by_split[split]]
        assert any(labels) and not all(labels), split
    assert ws.seed().threshold is None  # fitted on train by the run
    assert len(ws.pool()) == 10


def test_example_runs_end_to_end_offline(tmp_path):
    config = replace(load_classifier_config(_CONFIG), workspace=tmp_path / "ws")
    ws = Workspace(config)
    mismatch = {ex.state for ex in ws.examples if ex.label}

    # A fake model that reads mismatch questions well and capability questions badly.
    def answer(state: str, qid: str, q: Question) -> float:
        if "do not mention" in q.instructions or "surprised" in q.instructions:
            return round((0.8 if state in mismatch else 0.1) + jitter(state, qid), 2)
        return round(0.3 + 10 * jitter(state, qid), 2)

    result, ws = run_classifier(config, client=FakeSystemOne(answer, model="jev-latest"), log=lambda m: None)
    assert result.best_score > result.baseline_score
    assert (ws.dir / "report.md").exists()
    assert any("undisclosed" in q or "surprise" in q for q in result.best.score_ids)


def test_local_reflection_variant_loads_with_a_proposer():
    local = load_classifier_config(_CONFIG.with_name("config.lmstudio.yaml"))
    base = load_classifier_config(_CONFIG)
    assert local.proposer is not None and local.proposer.name == "lmstudio"
    assert local.splits == base.splits and local.dataset_path == base.dataset_path
    assert local.workspace != base.workspace  # its own cache and seal
