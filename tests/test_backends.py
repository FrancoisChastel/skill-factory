"""Cover optional-backend guards, the SkillOpt bridge, UI POST, and CLI evaluate."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.harness.callable_harness import CallableHarness
from skill_factory.metrics.golden import GoldenMetric, MatchMode
from skill_factory.optimizers.base import OptimizerConfig
from skill_factory.optimizers.skillopt import SkillOptOptimizer


def _echo_harness() -> CallableHarness:
    return CallableHarness(lambda skill, task: task.input)


def _dataset() -> Dataset:
    return Dataset([Task(id=f"t{i}", input=str(i), expected=str(i)) for i in range(4)])


# --- optional client import guards ---------------------------------------

def test_anthropic_client_requires_package():
    from skill_factory.llm.client import AnthropicClient

    if "anthropic" in sys.modules or _installed("anthropic"):
        pytest.skip("anthropic is installed in this environment")
    with pytest.raises(ImportError, match="anthropic"):
        AnthropicClient()


def test_openai_client_requires_package():
    from skill_factory.llm.openai_compat import OpenAICompatibleClient

    if _installed("openai"):
        pytest.skip("openai is installed in this environment")
    with pytest.raises(ImportError, match="openai"):
        OpenAICompatibleClient("some-model")


# --- SkillOpt bridge ------------------------------------------------------

def test_skillopt_requires_command():
    opt = SkillOptOptimizer()
    seed = Skill(name="s", description="d", body="b")
    train, val = _dataset().split(0.5, seed=0)
    with pytest.raises(ValueError, match="command"):
        opt.optimize(seed, train, val, _echo_harness(), GoldenMetric(MatchMode.EXACT))


def test_skillopt_runs_command_and_loads_output(tmp_path):
    seed = Skill(name="s", description="d", body="original body")
    train, val = _dataset().split(0.5, seed=0)
    # A stand-in "SkillOpt" command: copy the seed to best_skill.md in out_dir.
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; "
        "pathlib.Path('{out_dir}','best_skill.md').write_text(open('{seed}').read())",
    ]
    config = OptimizerConfig(params={"command": command, "work_dir": str(tmp_path / "so")})
    opt = SkillOptOptimizer(config=config)
    result = opt.optimize(seed, train, val, _echo_harness(), GoldenMetric(MatchMode.EXACT))

    assert result.optimizer == "skillopt"
    assert result.best_score == 1.0  # echo harness matches expected
    assert (tmp_path / "so" / "best_skill.md").exists()
    assert (tmp_path / "so" / "train.jsonl").exists()


def test_skillopt_missing_output_raises(tmp_path):
    seed = Skill(name="s", description="d", body="b")
    train, val = _dataset().split(0.5, seed=0)
    command = [sys.executable, "-c", "pass"]  # produces no best_skill.md
    config = OptimizerConfig(params={"command": command, "work_dir": str(tmp_path / "so2")})
    opt = SkillOptOptimizer(config=config)
    with pytest.raises(FileNotFoundError, match="did not produce"):
        opt.optimize(seed, train, val, _echo_harness(), GoldenMetric(MatchMode.EXACT))


# --- UI POST error paths --------------------------------------------------

def test_ui_post_and_404(tmp_path):
    from skill_factory.ui.server import _make_handler

    runs_root = (tmp_path / "runs").resolve()
    runs_root.mkdir(parents=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(runs_root))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        # missing 'config' -> 400
        req = urllib.request.Request(
            base + "/api/optimize", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400

        # unknown GET -> 404
        with pytest.raises(urllib.error.HTTPError) as exc2:
            urllib.request.urlopen(base + "/nope", timeout=5)
        assert exc2.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


# --- CLI evaluate ---------------------------------------------------------

def test_cli_evaluate(tmp_path, monkeypatch, capsys):
    from skill_factory import builder, cli

    (tmp_path / "seed.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    (tmp_path / "d.jsonl").write_text('{"id":"1","input":"a","expected":"a"}\n{"id":"2","input":"b","expected":"b"}\n')
    cfg = tmp_path / "c.yaml"
    cfg.write_text("skill: seed.md\ndataset: d.jsonl\n")

    from skill_factory.llm.client import FakeLLMClient

    monkeypatch.setattr(builder, "build_client", lambda provider: FakeLLMClient(responses=[""]))
    monkeypatch.setattr(builder, "build_harness", lambda c, cfg: _echo_harness())
    monkeypatch.setattr(builder, "build_metric", lambda mcfg, jc: GoldenMetric(MatchMode.EXACT))

    code = cli.main(["evaluate", "-c", str(cfg)])
    assert code == 0
    assert "Score: 1.000" in capsys.readouterr().out


def _installed(pkg: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(pkg) is not None


def _demo_result():
    from skill_factory.core.result import CandidateRecord, OptimizationResult

    seed = Skill(name="s", description="d", body="seed")
    best = seed.with_body("best")
    return OptimizationResult(
        best_skill=best, baseline_score=0.1, best_score=0.9,
        history=[CandidateRecord(0, seed, 0.1, 0.1, True, "seed")],
        optimizer="llm_loop",
    )


def test_ui_optimize_success(tmp_path, monkeypatch):
    from skill_factory import builder
    from skill_factory.ui.server import _make_handler

    (tmp_path / "c.yaml").write_text("skill: s.md\ndataset: d.jsonl\n")
    monkeypatch.setattr(builder, "run_optimization", lambda config: _demo_result())

    runs_root = (tmp_path / "runs").resolve()
    runs_root.mkdir(parents=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(runs_root))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = json.dumps({"config": str(tmp_path / "c.yaml"), "out": str(runs_root / "r")}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/optimize", data=body,
            headers={"Content-Type": "application/json"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert data["ok"] is True
        assert data["best_score"] == 0.9
    finally:
        server.shutdown()
        server.server_close()


def test_ui_main_entrypoint(monkeypatch):
    from skill_factory.ui import __main__ as ui_main

    called = {}
    monkeypatch.setattr(ui_main, "serve", lambda **kw: called.update(kw))
    monkeypatch.setattr(sys, "argv", ["prog", "--port", "9999", "--runs", "xyz"])
    ui_main.main()
    assert called["port"] == 9999
    assert called["runs_dir"] == "xyz"


def test_dspy_optimizer_import_guard():
    from skill_factory.optimizers.dspy_gepa import DspyGepaOptimizer

    if _installed("dspy"):
        pytest.skip("dspy is installed in this environment")
    with pytest.raises(ImportError, match="DSPy"):
        DspyGepaOptimizer()


def test_build_optimizer_skillopt_branch():
    from skill_factory.builder import build_optimizer
    from skill_factory.config import ProviderConfig

    opt = build_optimizer(
        {"name": "skillopt"}, optimizer_client=None, optimizer_provider=ProviderConfig()
    )
    assert opt.name == "skillopt"


def test_build_optimizer_dspy_branch_without_dspy():
    from skill_factory.builder import build_optimizer
    from skill_factory.config import ProviderConfig

    if _installed("dspy"):
        pytest.skip("dspy is installed in this environment")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(
            {"name": "dspy_gepa"}, optimizer_client=None,
            optimizer_provider=ProviderConfig(name="anthropic", model="m"),
        )


def test_run_optimization_with_judge_metric(tmp_path, monkeypatch):
    from skill_factory import builder
    from skill_factory.config import load_config
    from skill_factory.llm.client import FakeLLMClient

    (tmp_path / "seed.md").write_text("---\nname: s\ndescription: d\n---\nAnswer.\n")
    (tmp_path / "d.jsonl").write_text(
        "\n".join(f'{{"id":"t{i}","input":"{i}","expected":"{i}"}}' for i in range(6))
    )
    (tmp_path / "c.yaml").write_text(
        "skill: seed.md\ndataset: d.jsonl\nval_fraction: 0.5\n"
        "optimizer:\n  name: llm_loop\n  rounds: 1\n  minibatch_size: 2\n"
        "metric:\n  judge:\n    criteria:\n      - {name: correctness, description: x}\n"
    )
    judge_json = '{"scores": {"correctness": 10}, "rationale": "ok"}'
    monkeypatch.setattr(builder, "build_client", lambda provider: FakeLLMClient(responses=[judge_json]))
    monkeypatch.setattr(builder, "build_harness", lambda c, cfg: _echo_harness())

    config = load_config(tmp_path / "c.yaml")
    result = builder.run_optimization(config)
    assert result.best_score == 1.0  # judge always returns perfect
