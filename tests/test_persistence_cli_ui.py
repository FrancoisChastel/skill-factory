"""Tests for persistence, the CLI, and the UI server (all offline)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from skill_factory.cli import main
from skill_factory.core.result import CandidateRecord, OptimizationResult
from skill_factory.core.skill import Skill
from skill_factory.persistence import list_runs, load_run, save_run


def _result() -> OptimizationResult:
    seed = Skill(name="demo", description="d", body="seed body")
    best = seed.with_body("better body STRUCTURED")
    history = [
        CandidateRecord(0, seed, 0.2, 0.2, accepted=True, note="seed (baseline)"),
        CandidateRecord(1, best, 0.7, 0.8, accepted=True, note="edit"),
    ]
    return OptimizationResult(
        best_skill=best, baseline_score=0.2, best_score=0.8,
        history=history, optimizer="llm_loop",
    )


# --- persistence ----------------------------------------------------------

def test_save_and_load_run(tmp_path):
    artifacts = save_run(_result(), tmp_path / "run1")
    assert artifacts.best_skill.exists()
    assert artifacts.report.exists()
    payload = load_run(tmp_path / "run1")
    assert payload["optimizer"] == "llm_loop"
    assert payload["best_score"] == 0.8
    assert len(payload["history"]) == 2
    assert payload["best_skill"]["body"] == "better body STRUCTURED"


def test_list_runs_finds_all(tmp_path):
    save_run(_result(), tmp_path / "a")
    save_run(_result(), tmp_path / "b")
    runs = list_runs(tmp_path)
    assert {r["name"] for r in runs} == {"a", "b"}
    assert all(r["improvement"] == pytest.approx(0.6) for r in runs)


def test_list_runs_empty(tmp_path):
    assert list_runs(tmp_path / "missing") == []


# --- CLI ------------------------------------------------------------------

def test_cli_list_and_version(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "llm_loop" in out


def test_cli_export(tmp_path, capsys):
    skill_path = tmp_path / "SKILL.md"
    Skill(name="x", description="d", body="b").save(skill_path)
    code = main(["export", "--skill", str(skill_path), "--out", str(tmp_path / "dist")])
    assert code == 0
    assert (tmp_path / "dist" / "x" / "SKILL.md").exists()


def test_cli_optimize_end_to_end(tmp_path, monkeypatch, capsys):
    from skill_factory import builder

    # config file the CLI will load
    (tmp_path / "seed.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    (tmp_path / "d.jsonl").write_text('{"id":"1","input":"a","expected":"a"}\n{"id":"2","input":"b","expected":"b"}\n')
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"skill: seed.md\ndataset: d.jsonl\noutput:\n  dir: {tmp_path / 'out'}\n")

    monkeypatch.setattr(builder, "run_optimization", lambda config: _result())
    code = main(["optimize", "-c", str(cfg)])
    assert code == 0
    assert (tmp_path / "out" / "best_skill.md").exists()


def test_cli_error_returns_nonzero(capsys):
    code = main(["optimize", "-c", "/nonexistent/config.yaml"])
    assert code == 2


# --- UI server ------------------------------------------------------------

def test_ui_server_serves_runs_and_html(tmp_path):
    from skill_factory.ui.server import _make_handler

    runs_root = (tmp_path / "runs").resolve()
    runs_root.mkdir(parents=True)
    save_run(_result(), runs_root / "run-x")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(runs_root))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Skill Factory" in html

        runs = json.loads(urllib.request.urlopen(base + "/api/runs", timeout=5).read())
        assert runs["runs"][0]["name"] == "run-x"

        detail_url = base + "/api/run?dir=" + str(runs_root / "run-x")
        detail = json.loads(urllib.request.urlopen(detail_url, timeout=5).read())
        assert detail["optimizer"] == "llm_loop"
    finally:
        server.shutdown()
        server.server_close()
