"""Persist optimization runs to disk (for the CLI report and the UI)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skill_factory.core.result import OptimizationResult


@dataclass(frozen=True)
class RunArtifacts:
    """Paths written for a completed run."""

    run_dir: Path
    best_skill: Path
    report: Path
    result_json: Path


def save_run(result: OptimizationResult, out_dir: str | Path) -> RunArtifacts:
    """Write best_skill.md, report.md, and result.json into ``out_dir``."""
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_skill = run_dir / "best_skill.md"
    result.best_skill.save(best_skill)

    report = run_dir / "report.md"
    report.write_text(result.report(), encoding="utf-8")

    result_json = run_dir / "result.json"
    result_json.write_text(json.dumps(_to_payload(result), indent=2), encoding="utf-8")

    return RunArtifacts(run_dir, best_skill, report, result_json)


def load_run(run_dir: str | Path) -> dict:
    """Load a run's result.json payload."""
    p = Path(run_dir) / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"No result.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_runs(root: str | Path) -> list[dict]:
    """Return summaries of all runs under ``root`` (dirs containing result.json)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    runs = []
    for result_json in sorted(root_path.glob("**/result.json")):
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runs.append(
            {
                "dir": str(result_json.parent),
                "name": result_json.parent.name,
                "optimizer": payload.get("optimizer"),
                "baseline_score": payload.get("baseline_score"),
                "best_score": payload.get("best_score"),
                "improvement": payload.get("improvement"),
                "candidates": len(payload.get("history", [])),
            }
        )
    return runs


def _to_payload(result: OptimizationResult) -> dict:
    return {
        **result.to_dict(),
        "best_skill": {
            "name": result.best_skill.name,
            "description": result.best_skill.description,
            "body": result.best_skill.body,
            "markdown": result.best_skill.to_markdown(),
        },
        "history": [
            {
                "iteration": c.iteration,
                "train_score": round(c.train_score, 4),
                "val_score": round(c.val_score, 4),
                "accepted": c.accepted,
                "note": c.note,
                "skill_name": c.skill.name,
                "body": c.skill.body,
            }
            for c in result.history
        ],
    }
