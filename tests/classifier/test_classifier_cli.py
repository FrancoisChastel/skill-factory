"""The classifier mode from the command line, against a local System One stub over HTTP."""

from __future__ import annotations

import json

import pytest
import yaml

from skill_factory.classifier.config import is_classifier_config, load_classifier_config
from skill_factory.classifier.dataset import save_examples
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.cli import main

from .conftest import bank, make_examples

POLICY = '''
LEVELS = ("pass", "warn", "block")

def judge(example, answers, score, threshold):
    if score is not None and threshold is not None and score >= threshold:
        return "block" if score > 0.9 else "warn"
    return "pass"

def never(example, answers, score, threshold):
    return "pass"
'''


@pytest.fixture
def workspace(tmp_path, stub_server):
    url, stub = stub_server
    save_examples(make_examples(), tmp_path / "dataset.jsonl", positive="bad", negative="good")
    ProbeSet("seed", bank().subset(["capability"]), "capability", None, model="fake-jev",
             state_budget=10_000).save(tmp_path / "seed.json")
    bank().save(tmp_path / "pool.json")
    (tmp_path / "policy.py").write_text(POLICY)
    config = {
        "kind": "classifier",
        "name": "demo",
        "dataset": "dataset.jsonl",
        "labels": {"positive": "bad", "negative": "good"},
        "probeset": "seed.json",
        "pool": "pool.json",
        "workspace": "ws",
        "splits": {"held_out_sources": ["src-b"]},
        "systemone": {"provider": "custom", "base_url": url, "model": "fake-jev", "concurrency": 4,
                      "max_usd": 1.0},
        "objective": {"fpr_budget": 0.05, "constraints": [{"split": "val", "max_fpr": 0.1}]},
        "optimizer": {"rounds": 0},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path, stub


def run(*argv: str) -> int:
    return main(list(argv))


def test_config_detection_and_validation(workspace, tmp_path):
    root, _ = workspace
    assert is_classifier_config(root / "config.yaml")
    cfg = load_classifier_config(root / "config.yaml")
    assert cfg.positive == "bad" and cfg.workspace == root / "ws" and cfg.proposer is None
    (tmp_path / "bad.yaml").write_text("kind: classifier\ndataset: d\nprobeset: p\ntypo: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_classifier_config(tmp_path / "bad.yaml")
    (tmp_path / "skill.yaml").write_text("skill: s.md\ndataset: d\n")
    assert not is_classifier_config(tmp_path / "skill.yaml")
    assert not is_classifier_config(tmp_path / "missing.yaml")


def test_optimize_writes_the_workspace_and_keeps_test_sealed(workspace, capsys):
    root, stub = workspace
    assert run("optimize", "-c", str(root / "config.yaml")) == 0
    out = capsys.readouterr().out
    assert "classifier mode" in out and "sealed" in out
    ws = root / "ws"
    for name in ("answers.jsonl", "pool.json", "probeset.json", "report.md", "splits.svg", "history.json", "result.json"):
        assert (ws / name).exists(), name
    report = (ws / "report.md").read_text()
    assert "| test | sealed" in report and "| held-out | sealed" in report
    assert "## Rounds" in report and "## Questions" in report
    best = ProbeSet.load(ws / "probeset.json")
    assert best.name == "demo" and "intent@reviewer" in best.score_ids
    # Only train and val were ever sent to the model.
    test_states = {ex.state for ex in make_examples() if ex.source == "src-b"}
    assert not any(r["body"]["state"] in test_states for r in stub.requests)
    result = json.loads((ws / "result.json").read_text())
    assert result["kind"] == "classifier" and result["best_score"] > result["baseline_score"]


def test_second_run_is_free(workspace, capsys):
    root, stub = workspace
    run("optimize", "-c", str(root / "config.yaml"))
    sent = len(stub.requests)
    run("optimize", "-c", str(root / "config.yaml"))
    assert len(stub.requests) == sent  # everything came from the cache


def test_lab_commands_end_to_end(workspace, capsys):
    root, stub = workspace
    cfg = str(root / "config.yaml")
    assert run("lab", "ask", "-c", cfg, "--dry-run") == 0
    assert "requests to send" in capsys.readouterr().out and not stub.requests
    assert run("lab", "ask", "-c", cfg) == 0
    assert "answers cached" in capsys.readouterr().out
    assert run("lab", "status", "-c", cfg) == 0
    assert "lab online" in capsys.readouterr().out
    assert run("lab", "score", "-c", cfg, "--questions", "--failures", "train") == 0
    out = capsys.readouterr().out
    assert "intent@reviewer" in out and "SYSTEM-LEVEL CAUSES" in out
    assert run("lab", "select", "-c", cfg, "--out", str(root / "selected.json")) == 0
    assert "Pareto front" in capsys.readouterr().out and (root / "selected.json").exists()
    assert run("lab", "simulate", "-c", cfg, "--probeset", str(root / "selected.json"),
               "--policy", "policy.py:judge", "--policy", "policy.py:never") == 0
    out = capsys.readouterr().out
    assert "judge" in out and "block" in out and "never" in out


def test_sealed_splits_need_a_freeze_and_log_every_opening(workspace, capsys):
    root, _ = workspace
    cfg = str(root / "config.yaml")
    run("optimize", "-c", cfg)
    capsys.readouterr()
    assert run("lab", "report", "-c", cfg, "--splits", "test,held-out", "--ask") == 2
    assert "freeze" in capsys.readouterr().err
    assert run("lab", "score", "-c", cfg, "--splits", "test") == 2
    assert run("lab", "freeze", "-c", cfg, "--note", "final") == 0
    assert run("lab", "report", "-c", cfg, "--ask", "--out", str(root / "out")) == 0
    out = capsys.readouterr().out
    assert "test (" in out and "held-out (" in out and "opened 1 time" in out
    assert "-> **" in out  # the baseline is the seed fitted on train, not an unthresholded seed
    # Another candidate cannot be scored on the sealed splits through --before.
    assert run("lab", "report", "-c", cfg, "--before", str(root / "seed.json"), "--splits", "test") == 2
    assert "baseline" in capsys.readouterr().err
    assert (root / "out" / "report-splits.svg").exists()
    seal = json.loads((root / "ws" / "seal.json").read_text())
    assert seal["frozen"]["note"] == "final" and len(seal["views"]) == 1


def test_export_and_parity(workspace, capsys):
    root, _ = workspace
    cfg = str(root / "config.yaml")
    run("optimize", "-c", cfg)
    best = str(root / "ws" / "probeset.json")
    assert run("lab", "export", "--probeset", best, "--out", str(root / "probes.ts")) == 0
    assert run("lab", "parity", "--probeset", best, "--artifact", str(root / "probes.ts")) == 0
    assert "OK" in capsys.readouterr().out
    shipped = root / "probes.ts"
    shipped.write_text(shipped.read_text().replace("THRESHOLD = ", "THRESHOLD = 0.0 + "))
    assert run("lab", "parity", "--probeset", best, "--artifact", str(shipped)) == 4
    # Production verdicts identical to the lab's pass the behavior check.
    from skill_factory.classifier.workspace import Workspace

    ws = Workspace(load_classifier_config(cfg))
    ps = ProbeSet.load(best)
    table = ws.lab.table(ws.by_split["train"], ps.bank.compiled(), budget=ps.state_budget)
    rows = [{"id": ex.id, "flagged": ps.flags(table.of(ex.id))} for ex in ws.by_split["train"]]
    (root / "prod.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert run("lab", "parity", "--probeset", best, "--production", str(root / "prod.jsonl"), "-c", cfg) == 0
    assert run("lab", "parity", "--probeset", best) == 2
    assert run("lab", "parity", "--probeset", best, "--production", str(root / "prod.jsonl")) == 2
    seedless = ProbeSet.load(root / "seed.json")
    seedless.save(root / "nothreshold.json")
    assert run("lab", "export", "--probeset", str(root / "nothreshold.json"), "--out", str(root / "x.py")) == 2


def test_reask_and_recalibrate(workspace, capsys):
    root, _ = workspace
    cfg = str(root / "config.yaml")
    run("optimize", "-c", cfg)
    assert run("lab", "reask", "-c", cfg, "--split", "val") == 0
    assert "100% identical" in capsys.readouterr().out
    assert run("lab", "reask", "-c", cfg, "--split", "test") == 2
    assert run("lab", "recalibrate", "-c", cfg, "--model", "fake-jev-2", "--out", str(root / "recal.json")) == 0
    assert ProbeSet.load(root / "recal.json").model == "fake-jev-2"


def test_offline_lab_says_why(workspace, capsys, monkeypatch):
    root, _ = workspace
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["systemone"] = {"provider": "typesafe"}
    (root / "offline.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("SKILL_FACTORY_SYSTEMONE_KEY", raising=False)
    assert run("optimize", "-c", str(root / "offline.yaml")) == 2
    assert "no key for typesafe" in capsys.readouterr().err


def test_lab_without_subcommand_prints_help(capsys):
    assert run("lab") == 1


def test_failed_requests_reach_the_report_and_the_exit_code(workspace, capsys):
    root, stub = workspace
    stub.fail_next = [400, 400, 400]  # client errors are not retried
    assert run("optimize", "-c", str(root / "config.yaml")) == 3
    assert "3 System One requests failed" in capsys.readouterr().err
    report = (root / "ws" / "report.md").read_text()
    assert "WARNING: 3 requests failed" in report and "lower bound" in report
    assert json.loads((root / "ws" / "result.json").read_text())["metadata"]["failed_requests"] == 3
    # The next run fills the cache and is clean.
    assert run("optimize", "-c", str(root / "config.yaml")) == 0
