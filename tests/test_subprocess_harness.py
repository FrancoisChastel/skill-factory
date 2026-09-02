"""Tests for SubprocessHarness / ClaudeCodeHarness and their builder wiring.

All tests run real subprocesses, but only ``sys.executable`` one-liners — no
network, no external binaries.
"""

from __future__ import annotations

import sys

import pytest

from skill_factory.builder import build_harness, harness_needs_client
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task
from skill_factory.harness.subprocess_harness import ClaudeCodeHarness, SubprocessHarness

_SKILL = Skill(name="s", description="d", body="UPPERCASE the input.")
_TASK = Task(id="t1", input="hello world", expected="HELLO WORLD")

_PY = sys.executable


def test_stdin_mode_pipes_input():
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; print(sys.stdin.read().upper(), end='')"],
    )
    rollout = harness.run(_SKILL, _TASK)
    assert rollout.ok
    assert rollout.output == "HELLO WORLD"
    assert rollout.metadata["returncode"] == 0


def test_arg_mode_substitutes_input():
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; print(sys.argv[1][::-1])", "{input}"],
        input_via="arg",
    )
    rollout = harness.run(_SKILL, _TASK)
    assert rollout.ok
    assert rollout.output == "hello world"[::-1]


def test_skill_file_placeholder_written_and_readable():
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; print(open(sys.argv[1]).read())", "{skill_file}"],
    )
    rollout = harness.run(_SKILL, _TASK)
    assert rollout.ok
    assert "UPPERCASE the input." in rollout.output
    assert "name: s" in rollout.output  # full SKILL.md with frontmatter


def test_skill_body_placeholder_inline():
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; print(sys.argv[1])", "{skill_body}"],
    )
    rollout = harness.run(_SKILL, _TASK)
    assert rollout.output == "UPPERCASE the input."


def test_stray_braces_in_template_survive():
    # A template containing JSON braces must not crash substitution.
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; print(sys.argv[1])", '{"not": "a placeholder"}'],
    )
    rollout = harness.run(_SKILL, _TASK)
    assert rollout.output == '{"not": "a placeholder"}'


def test_nonzero_exit_is_failed_rollout():
    harness = SubprocessHarness(
        [_PY, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
    )
    rollout = harness.run(_SKILL, _TASK)
    assert not rollout.ok
    assert "exit 3" in (rollout.error or "")
    assert "boom" in (rollout.error or "")


def test_timeout_is_failed_rollout():
    harness = SubprocessHarness(
        [_PY, "-c", "import time; time.sleep(5)"],
        timeout=0.3,
    )
    rollout = harness.run(_SKILL, _TASK)
    assert not rollout.ok
    assert "timed out" in (rollout.error or "")


def test_missing_executable_is_failed_rollout():
    harness = SubprocessHarness(["/nonexistent/binary-xyz"])
    rollout = harness.run(_SKILL, _TASK)
    assert not rollout.ok
    assert "failed to launch" in (rollout.error or "")


def test_arg_mode_requires_input_placeholder():
    with pytest.raises(ValueError, match="input_via='arg'"):
        SubprocessHarness([_PY, "-c", "pass"], input_via="arg")


def test_claude_code_harness_argv():
    harness = ClaudeCodeHarness(model="claude-haiku-4-5", extra_args=["--max-turns", "1"])
    argv = harness.argv_template
    assert argv[0] == "claude"
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "text"
    assert argv[argv.index("--append-system-prompt") + 1] == "{skill_body}"
    assert argv[argv.index("--model") + 1] == "claude-haiku-4-5"
    assert argv[-2:] == ["--max-turns", "1"]
    assert harness.name == "claude_code"


# --- builder wiring -------------------------------------------------------

def test_build_harness_subprocess_type():
    harness = build_harness(
        {"type": "subprocess", "command": [_PY, "-c", "import sys; print(sys.stdin.read())"]},
    )
    assert isinstance(harness, SubprocessHarness)
    assert not harness_needs_client({"type": "subprocess"})


def test_build_harness_claude_code_type():
    harness = build_harness({"type": "claude_code", "model": "m", "executable": "claude"})
    assert isinstance(harness, ClaudeCodeHarness)
    assert not harness_needs_client({"type": "claude_code"})


def test_build_harness_api_requires_client():
    assert harness_needs_client({})  # default type is api
    with pytest.raises(ValueError, match="requires a target LLM client"):
        build_harness({"type": "api"}, None)


def test_build_harness_subprocess_requires_command():
    with pytest.raises(ValueError, match="command"):
        build_harness({"type": "subprocess"})


def test_build_harness_unknown_type():
    with pytest.raises(ValueError, match="Unknown harness type"):
        build_harness({"type": "teleport"})


def test_full_optimization_through_subprocess_harness(tmp_path, monkeypatch):
    """End-to-end: run_optimization with harness type=subprocess, no API client."""
    from skill_factory import builder
    from skill_factory.config import load_config

    (tmp_path / "seed.md").write_text("---\nname: s\ndescription: d\n---\nEcho.\n")
    (tmp_path / "d.jsonl").write_text(
        "\n".join(f'{{"id":"t{i}","input":"x{i}","expected":"x{i}"}}' for i in range(6))
    )
    py = sys.executable.replace("\\", "\\\\")
    # optimizer name != llm_loop so no optimizer client is built either
    # (build_optimizer is stubbed below) — then build_client must never fire.
    (tmp_path / "c.yaml").write_text(
        "skill: seed.md\ndataset: d.jsonl\nval_fraction: 0.5\n"
        "optimizer: {name: skillopt}\n"
        "metric: {golden: {mode: exact}}\n"
        f"harness:\n  type: subprocess\n  command: ['{py}', '-c', 'import sys; print(sys.stdin.read(), end=\"\")']\n"
    )

    # If run_optimization wrongly builds any LLM client for a subprocess
    # harness + golden metric, this sentinel makes the test fail loudly.
    def _no_client(provider):
        raise AssertionError("no LLM client should be built for this config")

    monkeypatch.setattr(builder, "build_client", _no_client)
    from skill_factory.core.result import CandidateRecord, OptimizationResult
    from skill_factory.evaluate import evaluate_skill

    class _NoopOptimizer:
        name = "noop"

        def optimize(self, seed, trainset, valset, harness, metric):
            score = evaluate_skill(seed, valset, harness, metric).score
            return OptimizationResult(
                best_skill=seed, baseline_score=score, best_score=score,
                history=[CandidateRecord(0, seed, score, score, True, "seed")],
                optimizer=self.name,
            )

    monkeypatch.setattr(builder, "build_optimizer", lambda cfg, **kw: _NoopOptimizer())

    result = builder.run_optimization(load_config(tmp_path / "c.yaml"))
    assert result.baseline_score == 1.0  # echo matches expected exactly
