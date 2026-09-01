"""Microsoft SkillOpt optimizer adapter (subprocess bridge).

SkillOpt (``pip install skillopt``) trains a ``skill.md`` like a neural network
via a rollout -> reflect -> aggregate -> select -> update -> gate loop, and emits
a deployable ``best_skill.md``. It runs as its own process against an *env*
package + YAML config.

Rather than reimplement SkillOpt's env system, this adapter treats SkillOpt as
an external optimizer with a clear I/O contract:

    inputs   seed_skill.md, train.jsonl, valid.jsonl  (materialized here)
    command  a user-configured command that runs SkillOpt and writes best_skill.md
    output   best_skill.md                            (loaded + scored here)

The produced skill is then scored with the factory's own harness + metric so the
reported baseline/best are comparable with the other backends.

Configure via OptimizerConfig.params:
    command        list[str] or str template; placeholders {seed} {train} {val} {out_dir}
    work_dir       where inputs/outputs are materialized (default: ./runs/skillopt)
    output_skill   path to best_skill.md relative to work_dir (default: best_skill.md)
    timeout        subprocess timeout seconds (default: 3600)
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from skill_factory.core.result import CandidateRecord, OptimizationResult
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset
from skill_factory.evaluate import evaluate_skill
from skill_factory.harness.base import Harness
from skill_factory.metrics.base import Metric
from skill_factory.optimizers.base import OptimizerConfig, register_optimizer

_DEFAULT_WORK_DIR = "runs/skillopt"
_DEFAULT_OUTPUT = "best_skill.md"


class SkillOptOptimizer:
    """Bridge to Microsoft SkillOpt via a configurable subprocess command."""

    name = "skillopt"

    def __init__(self, *, config: OptimizerConfig | None = None):
        self._config = config or OptimizerConfig()

    def optimize(
        self,
        seed: Skill,
        trainset: Dataset,
        valset: Dataset,
        harness: Harness,
        metric: Metric,
    ) -> OptimizationResult:
        params = self._config.params
        command = params.get("command")
        if not command:
            raise ValueError(
                "SkillOptOptimizer requires a 'command' in optimizer params that runs "
                "SkillOpt and writes best_skill.md. Example:\n"
                "  params:\n"
                "    command: 'python scripts/train.py --seed {seed} --train {train} "
                "--val {val} --out_root {out_dir}'\n"
                "See https://github.com/microsoft/SkillOpt for env/config setup."
            )

        work_dir = Path(params.get("work_dir", _DEFAULT_WORK_DIR))
        work_dir.mkdir(parents=True, exist_ok=True)
        seed_path = work_dir / "seed_skill.md"
        train_path = work_dir / "train.jsonl"
        val_path = work_dir / "valid.jsonl"
        seed.save(seed_path)
        _write_jsonl(train_path, trainset)
        _write_jsonl(val_path, valset)

        rendered = _render_command(
            command,
            seed=str(seed_path),
            train=str(train_path),
            val=str(val_path),
            out_dir=str(work_dir),
        )
        timeout = float(params.get("timeout", 3600))
        completed = subprocess.run(
            rendered,
            cwd=params.get("cwd"),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SkillOpt command failed (exit {completed.returncode}).\n"
                f"stderr:\n{completed.stderr[-2000:]}"
            )

        output_path = work_dir / params.get("output_skill", _DEFAULT_OUTPUT)
        if not output_path.exists():
            raise FileNotFoundError(
                f"SkillOpt did not produce an output skill at {output_path}. "
                "Set params.output_skill to the correct best_skill.md path."
            )

        candidate = seed.with_body(_load_body(output_path))

        workers = self._config.max_workers
        baseline = evaluate_skill(seed, valset, harness, metric, max_workers=workers).score
        cand_score = evaluate_skill(candidate, valset, harness, metric, max_workers=workers).score
        accepted = cand_score >= baseline
        best_skill = candidate if accepted else seed
        best_score = cand_score if accepted else baseline

        history = [
            CandidateRecord(0, seed, baseline, baseline, accepted=True, note="seed (baseline)"),
            CandidateRecord(1, candidate, cand_score, cand_score, accepted=accepted,
                            note="SkillOpt best_skill.md"),
        ]
        return OptimizationResult(
            best_skill=best_skill,
            baseline_score=baseline,
            best_score=best_score,
            history=history,
            optimizer=self.name,
            metadata={"backend": "skillopt", "work_dir": str(work_dir),
                      "output_skill": str(output_path)},
        )


def _write_jsonl(path: Path, dataset: Dataset) -> None:
    lines = []
    for task in dataset:
        record = {"id": task.id, "input": task.input}
        if task.expected is not None:
            record["expected"] = task.expected
        if task.metadata:
            record["metadata"] = dict(task.metadata)
        lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_command(command, **placeholders: str) -> list[str]:
    if isinstance(command, str):
        rendered = command.format(**placeholders)
        return shlex.split(rendered)
    return [str(part).format(**placeholders) for part in command]


def _load_body(path: Path) -> str:
    """Load a SkillOpt output skill. It may or may not carry YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    try:
        return Skill.from_markdown(text).body
    except ValueError:
        # No frontmatter — the whole file is the skill body.
        return text.strip()


register_optimizer("skillopt", lambda config, **kw: SkillOptOptimizer(config=config, **kw))
