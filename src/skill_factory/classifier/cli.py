"""``skill-factory lab <command>``: the classifier lab from the command line.

    ask           ask questions about every example, through the cache
    score         rescore a probe set offline: per split, per question, failures
    select        fit every candidate (single, mean, union, mean+lure) over the pool
    simulate      replay policies on cached answers, side by side
    freeze        seal the configuration; test and held-out can then be reported
    report        before/after table and figure on any splits (sealed ones need a freeze)
    reask         re-ask to measure determinism
    recalibrate   refit the threshold for another model
    export        write the probe set as JSON, TypeScript or Python
    parity        check a shipped artifact, or production verdicts, against the measured probe set
    import-scanner  turn a skill-scanner judge lab into a workspace
    status        splits, coverage, cache, seal, cost per pass
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skill_factory.classifier.config import ClassifierConfig, load_classifier_config


def register(sub: argparse._SubParsersAction) -> None:
    lab = sub.add_parser("lab", help="Classifier lab: cached probe-set tuning tools.")
    cmds = lab.add_subparsers(dest="lab_command")

    def cmd(name: str, help_text: str, handler, *, config: bool = True) -> argparse.ArgumentParser:
        p = cmds.add_parser(name, help=help_text)
        if config:
            p.add_argument("--config", "-c", required=True, help="classifier config YAML (kind: classifier)")
        p.set_defaults(handler=handler)
        return p

    p = cmd("ask", "Ask questions about every example of some splits, through the cache.", _ask)
    p.add_argument("--questions", help="question file (default: the probe set's questions and the pool)")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--budget", type=int, help="state budget (default: the probe set's)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and its estimated cost only")

    p = cmd("score", "Rescore a probe set offline.", _score)
    p.add_argument("--probeset", help="default: the workspace's best, else the seed")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--questions", action="store_true", help="per-question AUC and recall at the budget")
    p.add_argument("--failures", metavar="SPLIT", help="list misses and false flags with every P(true)")

    p = cmd("select", "Fit every candidate score over the question pool.", _select)
    p.add_argument("--fpr", type=float, help="false-positive budget (default: objective.fpr_budget)")
    p.add_argument("--max-questions", type=int, default=6)
    p.add_argument("--out", help="write the best candidate as a probe set here")

    p = cmd("simulate", "Replay policies on cached answers.", _simulate)
    p.add_argument("--policy", action="append", required=True, help="file.py:function (repeatable)")
    p.add_argument("--probeset")
    p.add_argument("--splits", default="train,val")

    p = cmd("freeze", "Seal the configuration so test and held-out can be reported.", _freeze)
    p.add_argument("--probeset", help="the configuration to report (default: the workspace's best)")
    p.add_argument("--baseline", help="what it is compared with (default: the seed, threshold fitted on train)")
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true", help="refreeze after sealed splits were opened (recorded)")

    p = cmd("report", "Before/after table and figure.", _report)
    p.add_argument("--before", help="probe set before (default: the seed, threshold fitted on train)")
    p.add_argument("--after", help="probe set after (default: the workspace's best)")
    p.add_argument("--splits", default="all",
                   help="'all' or a list; sealed splits need a freeze, and then only the frozen pair is scored")
    p.add_argument("--ask", action="store_true", help="ask missing answers first (costs money)")
    p.add_argument("--out", help="directory for report-splits.md and report-splits.svg")

    p = cmd("reask", "Re-ask a split to measure determinism.", _reask)
    p.add_argument("--probeset")
    p.add_argument("--split", default="val")
    p.add_argument("--replicate", type=int, default=1)

    p = cmd("recalibrate", "Refit the threshold for another model.", _recalibrate)
    p.add_argument("--model", required=True, help="the model to recalibrate for")
    p.add_argument("--probeset")
    p.add_argument("--fpr", type=float, help="default: the probe set's calibration.fpr_budget")
    p.add_argument("--out", required=True, help="where to write the recalibrated probe set")

    p = cmd("export", "Write a probe set as JSON, TypeScript or Python.", _export, config=False)
    p.add_argument("--probeset", required=True)
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--format", choices=("json", "ts", "py"))

    p = cmd("parity", "Check a shipped artifact or production verdicts against the measured probe set.",
            _parity, config=False)
    p.add_argument("--probeset", required=True, help="the measured probe set")
    p.add_argument("--artifact", help="the shipped file (.json, .ts, .py)")
    p.add_argument("--production", help="production verdicts JSONL: {id, verdict|flagged|score}")
    p.add_argument("--config", "-c", help="needed with --production")
    p.add_argument("--split", default="train")
    p.add_argument("--tolerance", type=float, default=0.02)

    p = cmd("import-scanner", "Turn a skill-scanner judge lab into a workspace.", _import_scanner, config=False)
    p.add_argument("lab_dir")
    p.add_argument("--out", "-o", required=True)
    p.add_argument("--budget", type=int, default=96_000)
    p.add_argument("--questions", help="the question file the lab asked (q-*.json)")

    cmd("status", "Splits, coverage, cache, seal and cost per pass.", _status)

    lab.set_defaults(handler=lambda args: (lab.print_help(), 1)[1])


# --- handlers ---------------------------------------------------------------

def _ws(args: argparse.Namespace):
    from skill_factory.classifier.workspace import Workspace

    return Workspace(load_classifier_config(args.config))


def _probeset(ws, path: str | None):
    from skill_factory.classifier.probeset import ProbeSet

    return ProbeSet.load(path) if path else ws.best()


def _splits(arg: str) -> list[str]:
    from skill_factory.classifier.splits import SPLITS

    names = list(SPLITS) if arg == "all" else [s.strip() for s in arg.split(",") if s.strip()]
    bad = [s for s in names if s not in SPLITS]
    if bad:
        raise ValueError(f"unknown split(s) {bad}; one of {', '.join(SPLITS)}")
    return names


def _refuse_sealed(names: list[str]) -> None:
    from skill_factory.classifier.splits import SEALED, SealedError

    if set(names) & SEALED:
        raise SealedError("this command reads tuning splits only (train, val); use `lab report` after `lab freeze`")


def _ask(args: argparse.Namespace) -> int:
    from skill_factory.classifier.questions import QuestionBank

    ws = _ws(args)
    names = _splits(args.splits)
    _refuse_sealed(names)
    probeset = ws.best()
    bank = QuestionBank.load(args.questions) if args.questions else ws.pool()
    budget = args.budget or probeset.state_budget
    examples = [ex for s in names for ex in ws.by_split[s]]
    questions = bank.compiled()
    jobs = ws.lab.plan(examples, questions, budget=budget)
    usd = ws.lab.settings.usd(ws.lab.estimate_tokens(jobs, questions))
    print(f"{len(examples)} examples, {len(questions)} questions: {len(jobs)} requests to send, about ${usd:.4f}")
    if args.dry_run or not jobs:
        return 0
    report = ws.lab.ask(examples, questions, budget=budget,
                        progress=lambda done, total: print(f"  {done}/{total}") if done % 200 == 0 else None)
    print(report.summary())
    for err in report.errors[:3]:
        print(f"  error: {err}")
    if args.questions:
        ws.pool().merge(bank).save(ws.path("pool.json"))
    return 0 if not report.failed else 3


def _score(args: argparse.Namespace) -> int:
    from skill_factory.classifier.drift import version_drift
    from skill_factory.classifier.evaluation import evaluate_split, question_stats
    from skill_factory.classifier.reflect import failure_digest
    from skill_factory.classifier.report import before_after_table, questions_table

    ws = _ws(args)
    names = _splits(args.splits)
    _refuse_sealed(names)
    probeset = _probeset(ws, args.probeset)
    examples = [ex for s in names for ex in ws.by_split[s]]
    policy = ws.policy()
    pool = ws.pool().merge(probeset.bank)
    table = ws.lab.table(examples, pool.compiled(), budget=probeset.state_budget)
    if (warning := version_drift(probeset, table)):
        print(f"warning: {warning}")
    reports = {s: evaluate_split(s, ws.by_split[s], table, probeset, policy) for s in names if ws.by_split[s]}
    print(before_after_table({}, reports))
    if args.questions:
        stats = question_stats(pool.ids, ws.by_split, table, ws.config.objective.fpr_budget, splits=names)
        print()
        print(questions_table(sorted(stats, key=lambda s: -s.auc.get("train", 0.5)), probeset.bank.ids,
                              ws.config.objective.fpr_budget, limit=200))
    if args.failures:
        _refuse_sealed([args.failures])
        print()
        print(failure_digest(probeset, ws.by_split[args.failures], table, policy=policy, max_examples=50,
                             excerpt_chars=300))
    return 0


def _select(args: argparse.Namespace) -> int:
    from skill_factory.classifier.evaluation import evaluate_split
    from skill_factory.classifier.select import candidates, pareto
    from skill_factory.classifier.report import pareto_table

    ws = _ws(args)
    probeset = ws.best()
    pool = ws.pool()
    fpr = args.fpr if args.fpr is not None else ws.config.objective.fpr_budget
    tuning = ws.by_split["train"] + ws.by_split["val"]
    table = ws.lab.table(tuning, pool.compiled(), budget=probeset.state_budget)
    cands = candidates(pool.ids, ws.by_split["train"], table, fpr, max_questions=args.max_questions)
    print(f"{len(cands)} candidates fitted on train at {fpr:.2%} false positives\n")
    print("| Candidate | Questions | Train | Val |\n|---|---:|---:|---:|")
    best = None
    for c in sorted(cands, key=lambda c: c.rank_key, reverse=True)[:15]:
        ps = probeset.with_changes(bank=pool.subset(list(c.questions)), score=c.score, threshold=c.threshold)
        v = evaluate_split("val", ws.by_split["val"], table, ps, ws.policy()).rates()
        print(f"| `{c.name}` | {len(c.questions)} | {c.train.recall:.1%} at {c.train.fpr:.2%} | "
              f"{v.recall:.1%} at {v.fpr:.2%} |")
        best = best or ps
    print("\nPareto front (train):\n" + pareto_table(pareto(cands)))
    if args.out and best is not None:
        print(f"\nwrote {best.save(args.out)}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from skill_factory.classifier.evaluation import evaluate_split

    ws = _ws(args)
    names = _splits(args.splits)
    _refuse_sealed(names)
    probeset = _probeset(ws, args.probeset)
    examples = [ex for s in names for ex in ws.by_split[s]]
    table = ws.lab.table(examples, ws.pool().merge(probeset.bank).compiled(), budget=probeset.state_budget)
    for ref in args.policy:
        policy = ws.policy(ref)
        cells = []
        for s in names:
            rep = evaluate_split(s, ws.by_split[s], table, probeset, policy)
            cells.append(f"{s}: " + ", ".join(
                f"{lvl} {r.tp}/{r.positives} ({r.recall:.1%}) fp {r.fp}/{r.negatives} ({r.fpr:.2%})"
                for lvl, r in rep.by_level.items()
            ))
        print(f"{policy.name:28s} " + "   ".join(cells))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    ws = _ws(args)
    from skill_factory.classifier.probeset import ProbeSet

    probeset = _probeset(ws, args.probeset)
    baseline = ProbeSet.load(args.baseline) if args.baseline else ws.baseline()
    seal = ws.seal()
    seal.freeze(probeset.content_hash(), baseline=baseline.content_hash(), note=args.note, force=args.force)
    print(f"frozen {probeset.name} as {probeset.content_hash()[:12]} against baseline "
          f"{baseline.content_hash()[:12]}; every report on test and held-out is now logged")
    return 0


def _report(args: argparse.Namespace) -> int:
    from skill_factory.classifier.probeset import ProbeSet
    from skill_factory.classifier.report import before_after_table, splits_svg

    ws = _ws(args)
    names = [s for s in _splits(args.splits) if ws.by_split.get(s)]
    before_ps = ProbeSet.load(args.before) if args.before else ws.baseline()
    after_ps = ProbeSet.load(args.after) if args.after else ws.best()
    before, after = ws.compare(before_ps, after_ps, names, ask=args.ask)
    table = before_after_table(before, after)
    print(table)
    print(f"\nseal: {ws.seal().summary()}")
    failed = sum(r.failed for r in ws.failed_asks)
    if failed:
        print(f"WARNING: {failed} requests failed; those examples count as misses above. Rerun with --ask.")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report-splits.md").write_text(table + "\n", encoding="utf-8")
        (out / "report-splits.svg").write_text(splits_svg(before, after), encoding="utf-8")
        print(f"wrote {out / 'report-splits.md'} and {out / 'report-splits.svg'}")
    return 3 if failed else 0


def _reask(args: argparse.Namespace) -> int:
    from skill_factory.classifier.drift import determinism

    ws = _ws(args)
    _refuse_sealed([args.split])
    probeset = _probeset(ws, args.probeset)
    result = determinism(ws.lab, probeset, ws.by_split[args.split], replicate=args.replicate)
    print(result.summary())
    for ex_id in result.flipped[:20]:
        print(f"  flipped: {ex_id}")
    return 3 if result.failed_requests else 0


def _recalibrate(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from skill_factory.classifier.drift import recalibrate
    from skill_factory.classifier.lab import Lab
    from skill_factory.classifier.workspace import build_systemone

    ws = _ws(args)
    probeset = _probeset(ws, args.probeset)
    client = build_systemone({**ws.config.systemone, "model": args.model})
    new_lab = Lab(client, ws.lab.cache, replace(ws.lab.settings), model=args.model)
    result = recalibrate(probeset, ws.lab, new_lab, ws.by_split, fpr_budget=args.fpr, policy=ws.policy())
    print(result.summary())
    print(f"wrote {result.new.save(args.out)}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from skill_factory.classifier.artifacts import write_export
    from skill_factory.classifier.probeset import ProbeSet

    probeset = ProbeSet.load(args.probeset)
    if probeset.threshold is None:
        raise ValueError("the probe set has no threshold yet; optimize or fit it before exporting")
    path = write_export(probeset, args.out, args.format)
    print(f"wrote {path} (sha256 {probeset.content_hash()[:12]})")
    return 0


def _parity(args: argparse.Namespace) -> int:
    from skill_factory.classifier.artifacts import check_artifact, check_production, load_production
    from skill_factory.classifier.evaluation import verdicts_for
    from skill_factory.classifier.probeset import ProbeSet

    probeset = ProbeSet.load(args.probeset)
    if not args.artifact and not args.production:
        raise ValueError("give --artifact and/or --production")
    ok = True
    if args.artifact:
        res = check_artifact(probeset, args.artifact)
        print(res.summary())
        ok = ok and res.ok
    if args.production:
        if not args.config:
            raise ValueError("--production needs --config (the dataset and the cached lab answers)")
        ws = _ws(args)
        _refuse_sealed([args.split])
        examples = ws.by_split[args.split]
        policy = ws.policy()
        table = ws.lab.table(examples, probeset.bank.compiled(), budget=probeset.state_budget)
        verdicts = verdicts_for(examples, table, probeset, policy)
        lab_detected = {k: policy.at_least(v, policy.levels[1]) for k, v in verdicts.items()}
        res = check_production(examples, lab_detected, load_production(args.production),
                               detect_levels=policy.detect_levels, threshold=probeset.threshold,
                               tolerance=args.tolerance)
        print(res.summary())
        for ex_id, lab_flag, prod_flag in res.disagreements[:20]:
            print(f"  {ex_id}: lab {'flags' if lab_flag else 'passes'}, production {'flags' if prod_flag else 'passes'}")
        ok = ok and res.ok
    return 0 if ok else 4


def _import_scanner(args: argparse.Namespace) -> int:
    from skill_factory.classifier.scanner_import import import_scanner_lab

    res = import_scanner_lab(args.lab_dir, args.out, budget=args.budget, questions=args.questions)
    print(f"imported {res.examples} examples into {res.dataset.parent}")
    print(f"  dataset  {res.dataset}\n  answers  {res.answers}" + (f"\n  pool     {res.pool}" if res.pool else ""))
    print("Write a config with `kind: classifier`, `dataset`, `probeset` and `workspace: .` to use it.")
    return 0


def _status(args: argparse.Namespace) -> int:
    from skill_factory.classifier.report import coverage_table

    ws = _ws(args)
    probeset = ws.best()
    cfg: ClassifierConfig = ws.config
    counts = {s: (sum(e.label for e in exs), len(exs) - sum(e.label for e in exs)) for s, exs in ws.by_split.items()}
    print(f"workspace {ws.dir}")
    print("splits: " + ", ".join(f"{s} {p}+/{n}-" for s, (p, n) in counts.items()))
    print(coverage_table(ws.statuses(probeset.state_budget)))
    print(f"cache: {len(ws.lab.cache)} answers; lab {'online' if ws.lab.client else 'offline: ' + ws.lab.offline_reason}")
    tuning = ws.by_split["train"] + ws.by_split["val"]
    print(f"probe set {probeset.name} ({probeset.content_hash()[:12]}), {len(probeset.bank)} questions; "
          f"pool {len(ws.pool())} questions")
    print(f"one pass of the probe set over train+val: about ${ws.lab.cost_per_pass(tuning, len(probeset.bank), budget=probeset.state_budget):.4f}; "
          f"of the pool: about ${ws.lab.cost_per_pass(tuning, len(ws.pool()), budget=probeset.state_budget):.4f}")
    print(f"seal: {ws.seal().summary()}")
    print(f"objective: {cfg.objective.split} recall at {cfg.objective.fpr_budget:.2%} false positives")
    return 0
