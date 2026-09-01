"""Command-line interface: ``skill-factory <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skill_factory import __version__


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill-factory",
        description="Scientifically optimize portable agent skills (SKILL.md).",
    )
    parser.add_argument("--version", action="version", version=f"skill-factory {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_opt = sub.add_parser("optimize", help="Run an optimization from a config file.")
    p_opt.add_argument("--config", "-c", required=True, help="Path to run config YAML.")
    p_opt.add_argument("--out", "-o", default=None, help="Output dir (default: config output.dir).")
    p_opt.add_argument("--emit", default=None, help="Also export best skill to this dir (npx skills layout).")
    p_opt.add_argument("--collection", action="store_true", help="Emit under skills/<slug>/.")
    p_opt.set_defaults(handler=_cmd_optimize)

    p_eval = sub.add_parser("evaluate", help="Evaluate a skill on a dataset (no optimization).")
    p_eval.add_argument("--config", "-c", required=True)
    p_eval.add_argument("--skill", default=None, help="Skill to evaluate (default: config skill).")
    p_eval.set_defaults(handler=_cmd_evaluate)

    p_exp = sub.add_parser("export", help="Export a SKILL.md into npx skills layout.")
    p_exp.add_argument("--skill", required=True)
    p_exp.add_argument("--out", "-o", required=True)
    p_exp.add_argument("--collection", action="store_true")
    p_exp.set_defaults(handler=_cmd_export)

    p_ui = sub.add_parser("ui", help="Launch the optional web dashboard.")
    p_ui.add_argument("--runs", default="runs", help="Runs directory to browse.")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.set_defaults(handler=_cmd_ui)

    p_list = sub.add_parser("list", help="List registered optimizers and providers.")
    p_list.set_defaults(handler=_cmd_list)

    return parser


def _cmd_optimize(args: argparse.Namespace) -> int:
    from skill_factory.builder import run_optimization
    from skill_factory.config import load_config
    from skill_factory.export import export_skill, install_hint
    from skill_factory.persistence import save_run

    config = load_config(args.config)
    print(f"Optimizing {config.skill_path.name} with '{config.optimizer.get('name', 'llm_loop')}'…")
    result = run_optimization(config)

    out_dir = args.out or config.output.get("dir", "runs/latest")
    artifacts = save_run(result, out_dir)

    print(result.report())
    print(f"Baseline {result.baseline_score:.3f} → best {result.best_score:.3f} "
          f"({result.improvement * 100:+.1f} pts)")
    print(f"Artifacts: {artifacts.run_dir}")

    emit_dir = args.emit or config.output.get("emit_skill_dir")
    if emit_dir:
        as_collection = args.collection or bool(config.output.get("as_collection", False))
        export_result = export_skill(result.best_skill, emit_dir, as_collection=as_collection)
        print(install_hint(export_result))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from skill_factory.builder import build_client, build_harness, build_metric
    from skill_factory.config import load_config
    from skill_factory.core.skill import Skill
    from skill_factory.core.task import Dataset
    from skill_factory.evaluate import evaluate_skill

    config = load_config(args.config)
    skill = Skill.load(args.skill) if args.skill else Skill.load(config.skill_path)
    dataset = Dataset.from_jsonl(config.dataset_path)

    target_client = build_client(config.target)
    harness = build_harness(target_client, config.harness)
    judge_client = build_client(config.judge_provider) if "judge" in config.metric else None
    metric = build_metric(config.metric, judge_client)

    evaluation = evaluate_skill(skill, dataset, harness, metric,
                                max_workers=int(config.harness.get("max_workers", 1)))
    print(f"Score: {evaluation.score:.3f} over {evaluation.size} tasks")
    print("\nWorst tasks:")
    print(evaluation.feedback_digest())
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from skill_factory.core.skill import Skill
    from skill_factory.export import export_skill, install_hint

    skill = Skill.load(args.skill)
    result = export_skill(skill, args.out, as_collection=args.collection)
    print(install_hint(result))
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from skill_factory.ui.server import serve

    serve(runs_dir=args.runs, host=args.host, port=args.port)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from skill_factory.llm.factory import SUPPORTED_PROVIDERS
    from skill_factory.optimizers.base import available_optimizers

    print("Optimizers:", ", ".join(available_optimizers()))
    print("Providers: ", ", ".join(SUPPORTED_PROVIDERS))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
