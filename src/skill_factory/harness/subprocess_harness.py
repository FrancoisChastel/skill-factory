"""Subprocess harnesses: run a skill through a real agent CLI.

This is the *in situ* evaluation mode: instead of simulating the skill as a
system prompt over a raw API, the skill is executed by the same binary it will
deploy to (Claude Code, Codex, any CLI). SkillOpt's paper shows skills optimized
inside their target harness transfer best — this makes that mode first-class.

Two classes:

* :class:`SubprocessHarness` — generic. You give an argv template with
  placeholders; each rollout renders it, runs the command, and captures stdout.
* :class:`ClaudeCodeHarness` — preset for the Claude Code CLI
  (``claude -p --append-system-prompt <skill> --output-format text``).

Placeholders available in argv templates:

    {skill_body}   the skill's markdown body, inline
    {skill_file}   path to a temp SKILL.md written for this rollout
    {input}        the task input, inline
    {input_file}   path to a temp file containing the task input

Task input is piped to stdin when ``input_via="stdin"`` (the default), which
avoids argv length limits and shell-quoting hazards.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task

_INPUT_TOKENS = ("{input}", "{input_file}")
_STDERR_TAIL = 1000


class SubprocessHarness:
    """Run each rollout as an external command built from an argv template."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        input_via: str = "stdin",
        timeout: float = 120.0,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        name: str = "subprocess",
    ):
        if not argv:
            raise ValueError("SubprocessHarness needs a non-empty argv template")
        if input_via not in ("stdin", "arg"):
            raise ValueError("input_via must be 'stdin' or 'arg'")
        joined = " ".join(str(a) for a in argv)
        if input_via == "arg" and not any(tok in joined for tok in _INPUT_TOKENS):
            raise ValueError(
                "input_via='arg' requires {input} or {input_file} in the argv template"
            )
        self._argv = [str(a) for a in argv]
        self._input_via = input_via
        self._timeout = timeout
        self._cwd = cwd
        self._env = dict(env) if env else None
        self.name = name

    @property
    def argv_template(self) -> list[str]:
        return list(self._argv)

    def run(self, skill: Skill, task: Task) -> Rollout:
        with tempfile.TemporaryDirectory(prefix="skill-factory-") as td:
            mapping = self._build_mapping(skill, task, Path(td))
            argv = [_substitute(part, mapping) for part in self._argv]
            stdin_data = task.input if self._input_via == "stdin" else None
            env = {**os.environ, **self._env} if self._env else None
            try:
                completed = subprocess.run(
                    argv,
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=self._cwd,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return Rollout.failed(task, f"command timed out after {self._timeout}s")
            except (OSError, ValueError) as exc:
                return Rollout.failed(task, f"failed to launch {argv[0]!r}: {exc}")

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()[-_STDERR_TAIL:]
            return Rollout.failed(
                task, f"exit {completed.returncode}: {stderr or '(no stderr)'}"
            )
        return Rollout(
            task=task,
            output=(completed.stdout or "").strip(),
            metadata={"harness": self.name, "returncode": completed.returncode},
        )

    def _build_mapping(self, skill: Skill, task: Task, tmp: Path) -> dict[str, str]:
        joined = " ".join(self._argv)
        mapping = {"{skill_body}": skill.body, "{input}": task.input}
        if "{skill_file}" in joined:
            skill_path = tmp / "SKILL.md"
            skill.save(skill_path)
            mapping["{skill_file}"] = str(skill_path)
        if "{input_file}" in joined:
            input_path = tmp / "input.txt"
            input_path.write_text(task.input, encoding="utf-8")
            mapping["{input_file}"] = str(input_path)
        return mapping


class ClaudeCodeHarness(SubprocessHarness):
    """Run the skill through the Claude Code CLI in print mode.

    Each rollout executes::

        claude -p --output-format text --append-system-prompt <skill body> \\
               [--model <model>] [extra args...]   < task input

    so the skill is evaluated by the exact harness it will deploy to. Requires
    the ``claude`` binary on PATH (or pass ``executable``).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        executable: str = "claude",
        timeout: float = 300.0,
        extra_args: Sequence[str] = (),
    ):
        argv = [
            executable,
            "-p",
            "--output-format",
            "text",
            "--append-system-prompt",
            "{skill_body}",
        ]
        if model:
            argv += ["--model", model]
        argv += list(extra_args)
        super().__init__(argv, input_via="stdin", timeout=timeout, name="claude_code")


def _substitute(part: str, mapping: Mapping[str, str]) -> str:
    """Replace known tokens only — stray braces in templates stay untouched."""
    for token, value in mapping.items():
        part = part.replace(token, value)
    return part
